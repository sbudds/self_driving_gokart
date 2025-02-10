import cv2
import torch
import serial
import time
import numpy as np
import multiprocessing
from multiprocessing import Process, Queue
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
        self.set_point = set_point  # target error = 0 (centered)
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

# --- Stop Sign Detection Process ---
def stop_sign_detection_process(frame_queue, serial_port, stop_interval, stop_duration):
    # Load YOLO model in this subprocess.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    TARGET_CLASSES = ['stop sign']
    last_stop_sign_time = 0

    # Initialize serial connection for stop sign commands.
    try:
        arduino = serial.Serial(serial_port, 9600, timeout=1)
        time.sleep(4)  # allow Arduino to initialize
        print(f"[StopSignProc] Connected to Arduino on {serial_port}")
    except Exception as e:
        print(f"[StopSignProc] Error connecting to Arduino: {e}")
        arduino = None

    while True:
        current_time = time.time()
        # Only process if enough time has passed since the last detection.
        if current_time - last_stop_sign_time < stop_interval:
            time.sleep(0.1)
            continue

        if frame_queue.empty():
            time.sleep(0.01)
            continue

        # Retrieve the latest frame from the queue.
        frame = frame_queue.get()

        try:
            resized_frame = cv2.resize(frame, (540, 260))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            continue

        results = yolo_model(rgb_frame)
        detections = results.xyxy[0]

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
            # Wait non-blockingly for STOP_DURATION seconds before sending GO.
            time.sleep(stop_duration)
            if arduino:
                try:
                    arduino.write(b'GO\n')
                    print("[StopSignProc] Sent GO to Arduino")
                except Exception as e:
                    print(f"[StopSignProc] Error sending GO: {e}")
        time.sleep(0.01)

# --- Main Process Setup ---
def main():
    # Initialize the lane detection model (UltrafastLaneDetector remains untouched)
    model_path = "lanes/models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

    # Video capture setup.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Unable to access video source.")
        return
    cv2.namedWindow("Lane and Stop Sign Detection", cv2.WINDOW_NORMAL)

    # Initialize Arduino serial connection for steering.
    SERIAL_PORT = '/dev/ttyACM0'  # Update if needed.
    try:
        arduino = serial.Serial(SERIAL_PORT, 9600, timeout=1)
        time.sleep(4)
        print(f"[Main] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"[Main] Error connecting to Arduino: {e}")
        arduino = None

    # Create a multiprocessing Queue for frames to be processed by the YOLO process.
    frame_queue = Queue(maxsize=5)

    # Start the stop sign detection subprocess.
    stop_process = Process(
        target=stop_sign_detection_process,
        args=(frame_queue, SERIAL_PORT, STOP_DETECTION_INTERVAL, STOP_DURATION)
    )
    stop_process.daemon = True
    stop_process.start()

    # Initialize PID controller and variables for steering.
    pid = PIDController(KP, KI, KD)
    prev_steering_angle = STEERING_CENTER
    frame_width = lane_detector.cfg.img_w

    print("[Main] Processing video input... (Press 'q' to quit)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] Error: Unable to read frame.")
            break

        # Send a copy of the frame to the YOLO process if the queue is not full.
        if not frame_queue.full():
            frame_queue.put(frame.copy())

        # Lane detection and steering.
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

        # Smooth the steering angle to reduce jitter.
        steering_angle = int(
            (1 - STEERING_SMOOTHING) * new_steering_angle +
            STEERING_SMOOTHING * prev_steering_angle
        )
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
    # Set the multiprocessing start method to 'spawn' so CUDA initializes properly.
    multiprocessing.set_start_method('spawn')
    main()
