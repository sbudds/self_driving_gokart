import cv2
import torch
import serial
import time
import numpy as np
import threading
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# --- Constants and PID Parameters ---
STEERING_CENTER = 105   # Neutral steering angle
STEERING_LEFT   = 80    # Maximum left steering angle
STEERING_RIGHT  = 130   # Maximum right steering angle

# PID tuning parameters (adjust as needed)
KP = 0.1
KI = 0.005
KD = 0.05

# Smoothing factor for steering (0: no smoothing, 1: fully previous value)
STEERING_SMOOTHING = 0.2

# Stop sign detection timing parameters
STOP_DETECTION_INTERVAL = 5  # seconds between stop sign detections
STOP_DURATION = 2            # seconds to remain stopped

# --- PID Controller Class ---
class PIDController:
    def __init__(self, kp, ki, kd, set_point=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point  # target error = 0 (lane center aligned with frame center)
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

# --- Helper Function to Compute Lane Center Offset ---
def get_lane_center_offset(lanes_points, lanes_detected, frame_width):
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

# --- Global Variables for Stop Sign Detection ---
latest_frame_lock = threading.Lock()
latest_frame = None
last_stop_sign_time = 0

# --- Stop Sign Detection Thread Function ---
def stop_sign_detection_thread(yolo_model, arduino):
    global latest_frame, last_stop_sign_time
    TARGET_CLASSES = ['stop sign']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    while True:
        current_time = time.time()
        # Process only every STOP_DETECTION_INTERVAL seconds
        if current_time - last_stop_sign_time < STOP_DETECTION_INTERVAL:
            time.sleep(0.1)
            continue

        # Copy the latest frame safely
        with latest_frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        # Resize and convert color for YOLO model
        resized_frame = cv2.resize(frame_copy, (540, 260))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference (this can be heavy, so it runs in its own thread)
        results = yolo_model(rgb_frame)
        detections = results.xyxy[0]

        stop_sign_found = False
        for *box, confidence, cls in detections:
            class_name = results.names[int(cls)]
            if class_name in TARGET_CLASSES:
                stop_sign_found = True
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{class_name} ({confidence:.2f})",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                break  # process one detection at a time

        if stop_sign_found:
            last_stop_sign_time = time.time()
            if arduino:
                try:
                    arduino.write(b'STOP\n')
                    print("Signal sent to Arduino: STOP")
                except Exception as e:
                    print(f"Error sending STOP to Arduino: {e}")
            # Schedule sending "GO" after STOP_DURATION seconds without blocking the main loop
            threading.Timer(STOP_DURATION, send_go_command, args=[arduino]).start()

        # Short sleep to yield control
        time.sleep(0.1)

def send_go_command(arduino):
    if arduino:
        try:
            arduino.write(b'GO\n')
            print("Signal sent to Arduino: GO")
        except Exception as e:
            print(f"Error sending GO to Arduino: {e}")

# --- Initialize Models and Hardware ---

# Lane Detector (unchanged from before)
model_path = "lanes/models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()
cv2.namedWindow("Lane and Stop Sign Detection", cv2.WINDOW_NORMAL)

# Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'  # Update if needed
BAUD_RATE = 9600
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(4)  # wait for Arduino to initialize
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# YOLOv5 model for stop sign detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# --- Initialize PID Controller and Steering Variables ---
pid = PIDController(KP, KI, KD)
prev_steering_angle = STEERING_CENTER
frame_width = lane_detector.cfg.img_w

# --- Start the Stop Sign Detection Thread ---
stop_thread = threading.Thread(target=stop_sign_detection_thread, args=(yolo_model, arduino))
stop_thread.daemon = True
stop_thread.start()

print("Processing video input... (Press 'q' to quit)")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Update the global latest_frame (for stop sign detection) safely
    with latest_frame_lock:
        latest_frame = frame.copy()

    # Lane detection and steering
    output_img = lane_detector.detect_lanes(frame)
    lanes_points = lane_detector.lanes_points
    lanes_detected = lane_detector.lanes_detected

    offset = get_lane_center_offset(lanes_points, lanes_detected, frame_width)
    if offset is not None:
        correction = pid.update(offset)
        new_steering_angle = STEERING_CENTER + correction
    else:
        new_steering_angle = STEERING_CENTER
        pid.integral = 0

    # Apply smoothing and clamp the steering angle
    steering_angle = int((1 - STEERING_SMOOTHING) * new_steering_angle +
                         STEERING_SMOOTHING * prev_steering_angle)
    prev_steering_angle = steering_angle
    steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

    # Send steering command to Arduino
    if arduino:
        try:
            arduino.write(f'{steering_angle}\n'.encode())
            print(f"Steering angle sent: {steering_angle}")
        except Exception as e:
            print(f"Error sending steering angle to Arduino: {e}")

    # Display lane detection output
    cv2.imshow("Lane and Stop Sign Detection", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup on exit
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
