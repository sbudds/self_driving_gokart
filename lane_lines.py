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
    # Convert PIL image to numpy array (this step is on CPU, but minimal)
    img_np = np.array(pil_img)  # shape: (H, W, 3) in RGB
    img_tensor = torch.from_numpy(img_np).float().cuda() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # shape: (1,3,H,W)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(288, 800),
                                                 mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.squeeze(0)  # shape: (3, 288, 800)

def gpu_preprocess_frame(frame):
    """
    Convert an OpenCV BGR frame to an RGB tensor on GPU, resized to (260, 540) for YOLO.
    """
    # Convert from BGR to RGB by slicing; this is cheap.
    frame_rgb = frame[..., ::-1]
    img_tensor = torch.from_numpy(frame_rgb).float().cuda() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(260, 540),
                                                 mode='bilinear', align_corners=False)
    return img_tensor

def gpu_process_output(output, cfg):
    """
    Process the lane detector model's output entirely on GPU.
    This example closely mimics the original processing:
      - Flips the tensor,
      - Applies softmax,
      - Computes a weighted sum.
    """
    processed_output = torch.flip(output[0], dims=[1])
    prob = torch.nn.functional.softmax(processed_output[:-1, :, :], dim=0)
    idx = (torch.arange(cfg.griding_num, device=processed_output.device).float() + 1.0).view(-1, 1, 1)
    loc = torch.sum(prob * idx, dim=0)
    argmax_output = torch.argmax(processed_output, dim=0)
    loc[argmax_output == cfg.griding_num] = 0
    return loc

###########################
# PID CONTROLLER (lightweight)
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
    Compute the offset from the frame center using the lane points.
    (Since the number of points is small, this is done on CPU.)
    """
    available_points = []
    for idx, detected in enumerate(lanes_detected):
        if detected and lanes_points[idx]:
            lane_x_avg = np.mean([pt[0] for pt in lanes_points[idx]])
            available_points.append(lane_x_avg)
    if available_points:
        lane_center = np.mean(available_points)
        return lane_center - (frame_width / 2.0)
    else:
        return None

###########################
# STOP SIGN DETECTION PROCESS (GPU-based pre-processing)
###########################

def stop_sign_detection_process(frame_queue, serial_port, stop_interval, stop_duration):
    # Load YOLOv5 model on GPU in this subprocess.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    TARGET_CLASSES = ['stop sign']
    last_stop_sign_time = 0
    frame_count = 0

    try:
        arduino = serial.Serial(serial_port, 9600, timeout=1)
        time.sleep(4)  # allow Arduino to initialize
        print(f"[StopSignProc] Connected to Arduino on {serial_port}")
    except Exception as e:
        print(f"[StopSignProc] Error connecting to Arduino: {e}")
        arduino = None

    while True:
        frame_count += 1
        current_time = time.time()
        if current_time - last_stop_sign_time < stop_interval:
            time.sleep(0.1)
            continue

        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        try:
            input_tensor = gpu_preprocess_frame(frame)
        except Exception as e:
            continue

        with torch.no_grad():
            results = yolo_model(input_tensor)
        # Minimal data transfer to CPU for detection loop:
        detections = results.xyxy[0].detach().cpu().numpy()

        stop_sign_found = False
        for *box, confidence, cls in detections:
            class_name = results.names[int(cls)]
            if class_name in TARGET_CLASSES:
                stop_sign_found = True
                print(f"[StopSignProc] Stop sign detected (conf: {confidence:.2f})")
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
        if frame_count % 100 == 0:
            torch.cuda.empty_cache()
        time.sleep(0.01)

###########################
# MAIN PROCESS SETUP
###########################

def main():
    print("[Main] Waiting 7 seconds for system initialization...")
    time.sleep(7)
    
    # Initialize lane detection model.
    model_path = "lanes/models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    
    # Override the image transform and output processing with our GPU-based versions.
    lane_detector.img_transform = gpu_image_transform_pil
    lane_detector.process_output = gpu_process_output

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Unable to access video source.")
        return
    cv2.namedWindow("Lane and Stop Sign Detection", cv2.WINDOW_NORMAL)

    SERIAL_PORT = '/dev/ttyACM0'  # Update if needed.
    try:
        arduino = serial.Serial(SERIAL_PORT, 9600, timeout=1)
        time.sleep(4)
        print(f"[Main] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"[Main] Error connecting to Arduino: {e}")
        arduino = None

    frame_queue = Queue(maxsize=5)
    stop_process = Process(
        target=stop_sign_detection_process,
        args=(frame_queue, SERIAL_PORT, 5, 2)
    )
    stop_process.daemon = True
    stop_process.start()

    pid = PIDController(0.1, 0.005, 0.05)
    STEERING_CENTER = 105
    STEERING_LEFT = 80
    STEERING_RIGHT = 130
    prev_steering_angle = STEERING_CENTER
    frame_width = lane_detector.cfg.img_w
    frame_count = 0

    print("[Main] Processing video input... (Press 'q' to quit)")
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("[Main] Error: Unable to read frame.")
            break

        if not frame_queue.full():
            frame_queue.put(frame.copy())

        # Run lane detection.
        output_img = lane_detector.detect_lanes(frame)
        # (Assuming lane_detector.lanes_points and lanes_detected are updated accordingly)
        try:
            lanes_points = lane_detector.lanes_points
            lanes_detected = lane_detector.lanes_detected
        except Exception as e:
            lanes_points = [[], [], [], []]
            lanes_detected = [False, False, False, False]

        offset = get_lane_center_offset(lanes_points, lanes_detected, frame_width)
        if offset is not None:
            correction = pid.update(offset)
            new_steering_angle = STEERING_CENTER + correction
        else:
            new_steering_angle = STEERING_CENTER
            pid.integral = 0

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

        if frame_count % 100 == 0:
            torch.cuda.empty_cache()

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
