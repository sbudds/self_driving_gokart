import cv2
import torch
import serial
import time
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

###########################
# GPU-BASED HELPER FUNCTIONS
###########################

def gpu_image_transform_pil(pil_img):
    """
    Convert a PIL image (in RGB) to a normalized tensor on GPU.
    The lane detector expects an image of size (288, 800).
    """
    # Minimal conversion: PIL -> NumPy (on CPU) is unavoidable,
    # but all heavy processing happens on GPU.
    img_np = np.array(pil_img)  # shape: (H, W, 3) in RGB
    img_tensor = torch.from_numpy(img_np).float().cuda() / 255.0  # [0,1] float tensor on GPU
    # Change shape to (1, 3, H, W)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    # Resize on GPU to (288, 800)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(288, 800),
                                                 mode='bilinear', align_corners=False)
    # Normalize using standard mean and std (broadcasted on GPU)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.squeeze(0)  # return shape: (3, 288, 800)

def gpu_process_output(output, cfg):
    """
    Process the lane detector output entirely on GPU.
    This function mimics the original post-processing:
      - Flips the tensor,
      - Applies softmax,
      - Computes a weighted sum over lane grid indices.
    """
    processed_output = torch.flip(output[0], dims=[1])
    prob = torch.nn.functional.softmax(processed_output[:-1, :, :], dim=0)
    idx = (torch.arange(cfg.griding_num, device=processed_output.device).float() + 1.0).view(-1, 1, 1)
    loc = torch.sum(prob * idx, dim=0)
    argmax_output = torch.argmax(processed_output, dim=0)
    loc[argmax_output == cfg.griding_num] = 0
    return loc

###########################
# PID CONTROLLER (Lightweight)
###########################

class PIDController:
    def __init__(self, kp, ki, kd, set_point=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point  # Target: lane center exactly at frame center
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
    Compute the horizontal offset (in pixels) between the detected lane center
    and the center of the frame.
    """
    available_points = []
    for idx, detected in enumerate(lanes_detected):
        if detected and lanes_points[idx]:
            # Compute the average x-coordinate for this lane
            lane_x_avg = np.mean([pt[0] for pt in lanes_points[idx]])
            available_points.append(lane_x_avg)
    if available_points:
        lane_center = np.mean(available_points)
        return lane_center - (frame_width / 2.0)
    else:
        return None

###########################
# MAIN PROGRAM: LANE DETECTION & STEERING
###########################

def main():
    # Wait 7 seconds for system and GPU initialization.
    print("[Main] Waiting 7 seconds for system initialization...")
    time.sleep(7)

    # Initialize the lane detection model.
    model_path = "lanes/models/tusimple_18.pth"  # Update path if needed.
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

    # Override the image transformation and output processing with our GPU-based versions.
    lane_detector.img_transform = gpu_image_transform_pil
    lane_detector.process_output = gpu_process_output

    # Set up video capture.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Unable to access video source.")
        return
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    # Initialize Arduino connection for steering commands.
    SERIAL_PORT = '/dev/ttyACM0'  # Update port if necessary.
    try:
        arduino = serial.Serial(SERIAL_PORT, 9600, timeout=1)
        time.sleep(4)
        print(f"[Main] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"[Main] Error connecting to Arduino: {e}")
        arduino = None

    # Set up PID controller for steering.
    pid = PIDController(0.1, 0.005, 0.05)
    STEERING_CENTER = 105  # Neutral steering angle.
    STEERING_LEFT = 80     # Maximum left.
    STEERING_RIGHT = 130   # Maximum right.
    prev_steering_angle = STEERING_CENTER
    frame_width = lane_detector.cfg.img_w  # Expected frame width (e.g., 1280).

    print("[Main] Processing video input... (Press 'q' to quit)")
    frame_count = 0
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("[Main] Error: Unable to read frame.")
            break

        # Run lane detection on the current frame.
        with torch.no_grad():
            output_img = lane_detector.detect_lanes(frame)

        # Retrieve lane points and detection flags.
        try:
            lanes_points = lane_detector.lanes_points
            lanes_detected = lane_detector.lanes_detected
        except Exception as e:
            lanes_points = [[], [], [], []]
            lanes_detected = [False, False, False, False]

        # Compute the lane center offset (on CPU; very little data).
        offset = get_lane_center_offset(lanes_points, lanes_detected, frame_width)
        if offset is not None:
            correction = pid.update(offset)
            new_steering_angle = STEERING_CENTER + correction
        else:
            new_steering_angle = STEERING_CENTER
            pid.integral = 0

        # Apply smoothing to reduce jitter.
        STEERING_SMOOTHING = 0.2
        steering_angle = int((1 - STEERING_SMOOTHING) * new_steering_angle +
                             STEERING_SMOOTHING * prev_steering_angle)
        prev_steering_angle = steering_angle
        steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

        # Send the steering command to the Arduino.
        if arduino:
            try:
                arduino.write(f'{steering_angle}\n'.encode())
            except Exception as e:
                print(f"[Main] Error sending steering angle: {e}")

        # Display the lane detection output.
        cv2.imshow("Lane Detection", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optionally clear the CUDA cache every 100 frames.
        if frame_count % 100 == 0:
            torch.cuda.empty_cache()

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()
