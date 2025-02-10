import cv2
import torch
import serial
import time
import numpy as np
import multiprocessing
from multiprocessing import Process, Queue
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

###########################
# GPU-BASED HELPER FUNCTIONS
###########################

def gpu_image_transform_pil(pil_img):
    """
    Convert a PIL image (in RGB) to a normalized tensor entirely on GPU.
    The model expects input of size (288, 800).
    """
    # Convert PIL image to numpy array (this step is on CPU,
    # but it's a one-time conversion per frame; all heavy math is on GPU)
    img_np = np.array(pil_img)  # shape: (H, W, 3), in RGB
    # Convert to torch tensor and send to GPU
    img_tensor = torch.from_numpy(img_np).float().cuda() / 255.0  # now in [0, 1]
    # Rearrange dimensions to (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)  # shape: (3, H, W)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # shape: (1, 3, H, W)
    # Resize on GPU using bilinear interpolation
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(288, 800), mode='bilinear', align_corners=False)
    # Normalize using mean and std (broadcasted on GPU)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    # Remove batch dimension (the lane detector later adds it again)
    return img_tensor.squeeze(0)  # shape: (3, 288, 800)

def gpu_preprocess_frame(frame):
    """
    Convert a NumPy frame (from OpenCV, BGR) to a GPU tensor in RGB, resized to (260, 540)
    for YOLO stop sign detection.
    """
    # Swap channels from BGR to RGB without extra CPU work (just reindexing)
    frame_rgb = frame[..., ::-1]
    # Convert to tensor and move to GPU
    img_tensor = torch.from_numpy(frame_rgb).float().cuda() / 255.0  # shape: (H, W, 3)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, H, W)
    # Resize using torch (YOLO expects 260x540 as per the original code)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(260, 540), mode='bilinear', align_corners=False)
    return img_tensor  # stays on GPU

def gpu_process_output(output, cfg):
    """
    A GPU-based re-implementation of the lane detection output processing.
    Assumes `output` is a torch.Tensor already on GPU.
    """
    # output: shape [1, channels, num_rows, num_lanes]
    # Flip along the row dimension (equivalent to output[:, ::-1, :])
    processed_output = torch.flip(output[0], dims=[1])
    # Apply softmax along the first channel dimension (excluding the last row)
    prob = torch.nn.functional.softmax(processed_output[:-1, :, :], dim=0)
    # Create an index tensor [1, 2, ..., griding_num] on GPU
    idx = (torch.arange(cfg.griding_num, device=processed_output.device).float() + 1.0).view(-1, 1, 1)
    # Weighted sum: sum(prob * idx)
    loc = torch.sum(prob * idx, dim=0)
    # Determine argmax for each position along the channel dimension
    argmax_output = torch.argmax(processed_output, dim=0)
    # Where argmax equals griding_num, set loc to 0
    loc[argmax_output == cfg.griding_num] = 0
    # Note: `loc` is a GPU tensor containing the computed positions.
    return loc  # Still on GPU

###########################
# PID Controller (lightweight; CPU overhead negligible)
###########################

class PIDController:
    def __init__(self, kp, ki, kd, set_point=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point  # target error = 0
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.0:
            dt = 1e-3
        error = self.set_point - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        self.last_time = current_time
        return output

def get_lane_center_offset(lanes_points, lanes_detected, frame_width):
    """
    Compute the offset of the detected lane center relative to the frame center.
    Here we do a quick computation on CPU since the amount of data is very small.
    """
    available_points = []
    for idx, detected in enumerate(lanes_detected):
        if detected and lanes_points[idx]:
            # Average x coordinate from lane points (list of [x,y])
            lane_x_avg = np.mean([pt[0] for pt in lanes_points[idx]])
            available_points.append(lane_x_avg)
    if available_points:
        lane_center = np.mean(available_points)
        return lane_center - (frame_width / 2.0)
    else:
        return None

###########################
# STOP SIGN DETECTION PROCESS (Using GPU pre-processing)
###########################

def stop_sign_detection_process(frame_queue, serial_port, stop_interval, stop_duration):
    # Load YOLOv5 model on GPU in this subprocess
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    TARGET_CLASSES = ['stop sign']
    last_stop_sign_time = 0

    try:
        arduino = serial.Serial(serial_port, 9600, timeout=1)
        time.sleep(4)  # allow Arduino to initialize
        print(f"[StopSignProc] Connected to Arduino on {serial_port}")
    except Exception as e:
        print(f"[StopSignProc] Error connecting to Arduino: {e}")
        arduino = None

    while True:
        current_time = time.time()
        if current_time - last_stop_sign_time < stop_interval:
            time.sleep(0.1)
            continue

        if frame_queue.empty():
            time.sleep(0.01)
            continue

        # Get latest frame and pre-process it entirely on GPU
        frame = frame_queue.get()
        try:
            input_tensor = gpu_preprocess_frame(frame)  # remains on GPU
        except Exception as e:
            continue

        # Run YOLO inference (entirely on GPU)
        results = yolo_model(input_tensor)
        # Get detections (move to CPU only for the final minimal loop)
        detections = results.xyxy[0].detach().cpu().numpy()

        stop_sign_found = False
        for *box, confidence, cls in detections:
            class_name = results.names[int(cls)]
            if class_name in TARGET_CLASSES:
                stop_sign_found = True
                print(f"[StopSignProc] Stop sign detected with confidence: {confidence:.2f}")
                break

        if stop_sign_found:
            last_stop_sign_time = time.time()
            if arduino:
                try:
                    arduino.write(b'STOP\n')
                    print("[StopSignProc] Sent STOP to Arduino")
                except Exception as e:
                    print(f"[StopSignProc] Error sending STOP: {e}")
            time.sleep(stop_duration)
            if arduino:
                try:
                    arduino.write(b'GO\n')
                    print("[StopSignProc] Sent GO to Arduino")
                except Exception as e:
                    print(f"[StopSignProc] Error sending GO: {e}")
        time.sleep(0.01)

###########################
# MAIN PROCESS SETUP
###########################

def main():
    # Give the system 7 seconds to initialize (all CUDA contexts, etc.)
    print("[Main] Waiting 7 seconds for system initialization...")
    time.sleep(7)
    
    # Initialize the lane detection model (UltrafastLaneDetector remains untouched)
    model_path = "lanes/models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    
    # Override the image transform and output processing functions with GPU-based versions.
    lane_detector.img_transform = gpu_image_transform_pil
    lane_detector.process_output = gpu_process_output

    # Video capture setup (cannot be moved to GPU)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Unable to access video source.")
        return
    cv2.namedWindow("Lane and Stop Sign Detection", cv2.WINDOW_NORMAL)

    # Initialize Arduino serial for steering commands.
    SERIAL_PORT = '/dev/ttyACM0'  # Update as needed.
    try:
        arduino = serial.Serial(SERIAL_PORT, 9600, timeout=1)
        time.sleep(4)
        print(f"[Main] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"[Main] Error connecting to Arduino: {e}")
        arduino = None

    # Create a multiprocessing Queue for frames for YOLO processing.
    frame_queue = Queue(maxsize=5)

    # Start the stop sign detection subprocess.
    stop_process = Process(
        target=stop_sign_detection_process,
        args=(frame_queue, SERIAL_PORT, 5, 2)  # using STOP_DETECTION_INTERVAL=5, STOP_DURATION=2
    )
    stop_process.daemon = True
    stop_process.start()

    # Initialize PID controller for steering.
    pid = PIDController(0.1, 0.005, 0.05)
    STEERING_CENTER = 105
    STEERING_LEFT = 80
    STEERING_RIGHT = 130
    prev_steering_angle = STEERING_CENTER
    frame_width = lane_detector.cfg.img_w  # Expected image width from lane detector config

    print("[Main] Processing video input... (Press 'q' to quit)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] Error: Unable to read frame.")
            break

        # Send a copy of the frame to the YOLO process if the queue is not full.
        if not frame_queue.full():
            frame_queue.put(frame.copy())

        # Run lane detection. The input image is converted using our GPU-based transform.
        output_img = lane_detector.detect_lanes(frame)
        # The lane detector’s process_output now runs on GPU.
        # (For steering calculations and drawing, we need to bring minimal data back to CPU.)
        try:
            # Assume lanes_points and lanes_detected are computed inside the model.
            # (In the original code, lanes_points are built from the processed output.)
            # For demonstration, we simulate these values. In a real scenario, you might
            # convert the GPU tensor to a NumPy array only for the lightweight offset computation.
            lanes_points = lane_detector.lanes_points  # This may still be a GPU tensor
            lanes_detected = lane_detector.lanes_detected  # Likely a NumPy array (set by model code)
        except Exception as e:
            lanes_points = [[], [], [], []]
            lanes_detected = [False, False, False, False]

        # For steering, we transfer only the needed value to CPU.
        offset = get_lane_center_offset(lanes_points, lanes_detected, frame_width)
        if offset is not None:
            correction = pid.update(offset)
            new_steering_angle = STEERING_CENTER + correction
        else:
            new_steering_angle = STEERING_CENTER
            pid.integral = 0

        # Smooth the steering angle to reduce jitter.
        STEERING_SMOOTHING = 0.2
        steering_angle = int((1 - STEERING_SMOOTHING) * new_steering_angle +
                             STEERING_SMOOTHING * prev_steering_angle)
        prev_steering_angle = steering_angle
        steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

        if arduino:
            try:
                arduino.write(f'{steering_angle}\n'.encode())
            except Exception as e:
                print(f"[Main] Error sending steering angle: {e}")

        cv2.imshow("Lane and Stop Sign Detection", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    # Use the 'spawn' start method so that CUDA contexts are initialized properly in subprocesses.
    multiprocessing.set_start_method('spawn')
    main()
