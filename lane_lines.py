import cv2
import torch
import serial
import time
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# --- Constants and PID Parameters ---
STEERING_CENTER = 105   # Center (neutral) steering angle
STEERING_LEFT   = 80    # Maximum left steering angle
STEERING_RIGHT  = 130   # Maximum right steering angle

# PID tuning parameters (adjust these as needed)
KP = 0.1
KI = 0.005
KD = 0.05

# Smoothing factor for steering (0: no smoothing, 1: fully previous value)
STEERING_SMOOTHING = 0.2

# --- PID Controller Class ---
class PIDController:
    def __init__(self, kp, ki, kd, set_point=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point  # target error is zero (i.e. lane center exactly at frame center)
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.0:
            dt = 1e-3  # avoid division by zero
        error = self.set_point - measurement  # note: error is negative if lane center is to the right of frame center
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        self.last_time = current_time
        return output

# --- Helper Function to Compute Lane Center Offset ---
def get_lane_center_offset(lanes_points, lanes_detected, frame_width):
    """
    Compute the offset of the detected lane center relative to the center of the frame.
    Combines any available lanes (not just left/right) for robustness.
    Returns None if no reliable lane points are available.
    """
    available_points = []
    for idx, detected in enumerate(lanes_detected):
        if detected and len(lanes_points[idx]) > 0:
            # Use the average x coordinate of the lane’s points
            lane_x_avg = np.mean([pt[0] for pt in lanes_points[idx]])
            available_points.append(lane_x_avg)
    if len(available_points) > 0:
        lane_center = np.mean(available_points)
        frame_center = frame_width / 2.0
        return lane_center - frame_center
    else:
        return None

# --- Initialize Models and Hardware ---
# Lane Detector (the ultrafast lane detector module – no changes made here)
model_path = "lanes/models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Video capture (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()
cv2.namedWindow("Lane and Stop Sign Detection", cv2.WINDOW_NORMAL)

# Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'  # Update as needed
BAUD_RATE = 9600
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(4)  # wait for Arduino to initialize
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# YOLOv5 model for Stop Sign Detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
TARGET_CLASSES = ['stop sign']
last_stop_detection_time = 0  # used to throttle stop sign detection frequency

def detect_stop_signs(frame):
    global last_stop_detection_time
    current_time = time.time()
    # Run detection only every 5 seconds to prevent repeated signals
    if current_time - last_stop_detection_time < 5:
        return

    resized_frame = cv2.resize(frame, (540, 260))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(rgb_frame)
    detections = results.xyxy[0]  # bounding box detections

    for *box, confidence, cls in detections:
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Stop sign detected with confidence: {confidence:.2f}")
            if arduino:
                try:
                    arduino.write(b'STOP\n')
                    print("Signal sent to Arduino: STOP")
                    time.sleep(2)  # duration of stop
                    arduino.write(b'GO\n')
                    print("Signal sent to Arduino: GO")
                    last_stop_detection_time = time.time()
                except Exception as e:
                    print(f"Error sending signal to Arduino: {e}")
            break

# --- Initialize PID Controller and Variables for Steering ---
pid = PIDController(KP, KI, KD)
prev_steering_angle = STEERING_CENTER  # start at center
frame_width = lane_detector.cfg.img_w  # use expected width from lane detector config

print("Processing video input... (Press 'q' to quit)")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Run lane detection
    output_img = lane_detector.detect_lanes(frame)
    lanes_points = lane_detector.lanes_points
    lanes_detected = lane_detector.lanes_detected

    # Run stop sign detection (which draws boxes on frame and sends signals)
    detect_stop_signs(frame)

    # Compute the lane center offset
    offset = get_lane_center_offset(lanes_points, lanes_detected, frame_width)
    if offset is not None:
        # Use the PID controller to compute a correction based on the offset
        correction = pid.update(offset)
        # New steering angle: add correction (note: correction can be negative)
        new_steering_angle = STEERING_CENTER + correction
    else:
        # No reliable lane detection: gently revert to center
        new_steering_angle = STEERING_CENTER
        pid.integral = 0  # optionally reset the integrator

    # Smooth the steering angle to reduce jitter
    steering_angle = int((1 - STEERING_SMOOTHING) * new_steering_angle +
                         STEERING_SMOOTHING * prev_steering_angle)
    prev_steering_angle = steering_angle

    # Clamp the steering angle within the physical limits
    steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

    # Send the steering command to Arduino
    if arduino:
        try:
            arduino.write(f'{steering_angle}\n'.encode())
            print(f"Steering angle sent: {steering_angle}")
        except Exception as e:
            print(f"Error sending steering angle to Arduino: {e}")

    # Display the annotated frame (with lane markings and stop sign boxes)
    cv2.imshow("Lane and Stop Sign Detection", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup on exit
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
