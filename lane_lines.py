import cv2
import torch
import time
import numpy as np
import threading
import queue
import types
import Jetson.GPIO as GPIO
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

###########################
# GPIO/Servo SETUP (Using Jetson.GPIO)
###########################

# Define the servo control parameters.
# Typical hobby servos expect a pulse width of ~1ms to 2ms (5%-10% duty cycle at 50Hz).
# We assume STEERING_CENTER = 105, STEERING_LEFT = 80, STEERING_RIGHT = 130.
# Mapping: 105 -> 7.5% duty, 80 -> ~5%, 130 -> ~10%
SERVO_PIN = 12        # Update if needed.
PWM_FREQUENCY = 50    # 50 Hz (period of 20ms)

def angle_to_duty(angle):
    """Convert a steering angle into a PWM duty cycle."""
    # Center at 105 corresponds to 7.5% duty cycle.
    return 7.5 + (angle - 105) * 0.1

# Setup GPIO using BOARD numbering.
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)
STEERING_CENTER = 105
servo_pwm.start(angle_to_duty(STEERING_CENTER))


###########################
# GPU-BASED HELPER FUNCTIONS
###########################

def gpu_image_transform(frame):
    """
    Convert an OpenCV BGR frame to a normalized tensor on GPU.
    The lane detector expects an image of size (288, 800) in RGB.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(frame_rgb).float().cuda() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(288, 800),
                                                 mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.squeeze(0)

def gpu_process_output(output, cfg):
    """
    Process the lane detector model's output entirely on the GPU.
    Returns:
      - lanes_points: list of torch tensors (one per lane) with shape (N, 2)
      - lanes_detected: a boolean tensor (one per lane)
      - lane_center: scalar tensor for the overall average x–coordinate of detected lanes
    """
    with torch.no_grad():
        processed_output = torch.flip(output[0], dims=[1])
        prob = torch.nn.functional.softmax(processed_output[:-1, :, :], dim=0)
        idx = (torch.arange(cfg.griding_num, device=processed_output.device).float() + 1.0).view(-1, 1, 1)
        loc = torch.sum(prob * idx, dim=0)
        argmax_output = torch.argmax(processed_output, dim=0)
        loc[argmax_output == cfg.griding_num] = 0
        
        col_sample = torch.linspace(0, 800 - 1, steps=cfg.griding_num, device=processed_output.device)
        col_sample_w = col_sample[1] - col_sample[0] if cfg.griding_num > 1 else torch.tensor(0.0, device=processed_output.device)
        x_coords = loc * (col_sample_w * cfg.img_w / 800.0) - 1.0
        
        valid_mask = loc > 0
        valid_counts = valid_mask.sum(dim=0).float()
        sum_x = (x_coords * valid_mask.float()).sum(dim=0)
        avg_x = torch.where(valid_counts > 0, sum_x / valid_counts, torch.zeros_like(sum_x))
        lanes_detected = valid_counts > 2
        
        if lanes_detected.any():
            lane_center = avg_x[lanes_detected].mean()
        else:
            lane_center = torch.tensor(cfg.img_w / 2.0, device=processed_output.device)
        
        # Prepare lane points for visualization.
        row_anchor = torch.tensor(cfg.row_anchor, device=processed_output.device, dtype=torch.float32)
        y_coords = cfg.img_h * (row_anchor.flip(0) / 288.0) - 1.0
        
        lanes_points = []
        for lane in range(loc.shape[1]):
            valid_indices = valid_mask[:, lane]
            if valid_indices.any():
                lane_x = x_coords[:, lane][valid_indices]
                lane_y = y_coords[valid_indices]
                lane_points = torch.stack([lane_x, lane_y], dim=1)
            else:
                lane_points = torch.empty((0, 2), device=processed_output.device)
            lanes_points.append(lane_points)
        
        return lanes_points, lanes_detected, lane_center

###########################
# PID CONTROLLER (Lightweight)
###########################

class PIDController:
    def __init__(self, kp, ki, kd, set_point=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
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

###########################
# MAIN PROGRAM: LANE DETECTION & SERVO STEERING
###########################

def main():
    print("[Main] Waiting 7 seconds for system initialization...")
    time.sleep(7)

    model_path = "lanes/models/tusimple_18.pth"  # Update path as needed.
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    lane_detector.img_transform = gpu_image_transform
    lane_detector.process_output = gpu_process_output

    # Monkey-patch detect_lanes to store GPU results and draw lane lines.
    def detect_lanes_wrapper(self, frame):
        input_tensor = self.img_transform(frame).unsqueeze(0)
        output = self.model(input_tensor)
        lanes_points, lanes_detected, lane_center = self.process_output(output, self.cfg)
        self.lanes_points = lanes_points
        self.lanes_detected = lanes_detected
        self.lane_center = lane_center

        # Draw lane lines on the frame.
        lane_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for i, lane in enumerate(lanes_points):
            if lane.numel() > 0:
                lane_cpu = lane.detach().cpu().numpy()
                points = []
                for point in lane_cpu:
                    x, y = int(point[0]), int(point[1])
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, lane_colors[i % len(lane_colors)], -1)
                if len(points) >= 2:
                    cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False,
                                  color=lane_colors[i % len(lane_colors)], thickness=2)
        return frame

    lane_detector.detect_lanes = types.MethodType(detect_lanes_wrapper, lane_detector)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Unable to access video source.")
        return
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    pid = PIDController(0.1, 0.005, 0.05)
    STEERING_CENTER = 105
    STEERING_LEFT = 80
    STEERING_RIGHT = 130
    prev_steering_angle = STEERING_CENTER
    frame_width = lane_detector.cfg.img_w

    # PID and servo update parameters.
    PID_UPDATE_INTERVAL = 3           # Update PID every 3 frames.
    ANGLE_CHANGE_THRESHOLD = 2        # Only update servo if angle changes by >2°.
    NO_LANE_UPDATE_INTERVAL = 1.0       # When no lanes detected, update less frequently.
    MAX_UPDATE_INTERVAL = 0.5           # Force an update if >0.5 sec since last command.

    last_no_lane_time = time.time()
    last_sent_angle = STEERING_CENTER
    last_command_time = time.time()

    print("[Main] Processing video input... (Press 'q' to quit)")
    frame_count = 0
    last_lane_center = frame_width / 2.0
    last_offset = 0.0
    last_correction = 0.0
    last_steering_angle = STEERING_CENTER

    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("[Main] Error: Unable to read frame.")
            break

        # Process lane detection and draw lane lines.
        with torch.no_grad():
            output_img = lane_detector.detect_lanes(frame)

        # Update the PID and servo commands every PID_UPDATE_INTERVAL frames.
        if frame_count % PID_UPDATE_INTERVAL == 0:
            try:
                lane_center = lane_detector.lane_center
            except AttributeError:
                lane_center = torch.tensor(frame_width / 2.0, device='cuda')
            offset = (lane_center - torch.tensor(frame_width / 2.0, device=lane_center.device)).item()
            correction = pid.update(offset)
            new_steering_angle = STEERING_CENTER + correction

            STEERING_SMOOTHING = 0.2
            steering_angle = int((1 - STEERING_SMOOTHING) * new_steering_angle +
                                 STEERING_SMOOTHING * prev_steering_angle)
            steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))
            
            # Save values for visualization.
            last_lane_center = lane_center.item()
            last_offset = offset
            last_correction = correction
            last_steering_angle = steering_angle

            current_time = time.time()
            # Throttle updates:
            if lane_detector.lanes_detected.any():
                # When lanes are detected, update if the angle changes significantly or if too much time has passed.
                if (abs(steering_angle - last_sent_angle) >= ANGLE_CHANGE_THRESHOLD or 
                    (current_time - last_command_time) > MAX_UPDATE_INTERVAL):
                    duty = angle_to_duty(steering_angle)
                    servo_pwm.ChangeDutyCycle(duty)
                    last_sent_angle = steering_angle
                    last_command_time = current_time
            else:
                # When no lanes are detected, update at a slower rate with the neutral command.
                if current_time - last_no_lane_time > NO_LANE_UPDATE_INTERVAL:
                    duty = angle_to_duty(STEERING_CENTER)
                    servo_pwm.ChangeDutyCycle(duty)
                    last_sent_angle = STEERING_CENTER
                    last_no_lane_time = current_time
                    last_command_time = current_time

            prev_steering_angle = steering_angle

        # Overlay debugging text on the video feed.
        cv2.putText(output_img, f"Lane Center: {last_lane_center:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_img, f"Offset: {last_offset:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_img, f"PID Correction: {last_correction:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_img, f"Steering Angle: {last_steering_angle}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Lane Detection", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    servo_pwm.stop()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
