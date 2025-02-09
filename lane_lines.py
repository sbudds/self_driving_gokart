import cv2
import torch
import serial
import time
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# Model and hardware setup
model_path = "lanes/models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True  
weights_only = True

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

# Initialize Arduino serial connection
SERIAL_PORT = '/dev/ttyACM0'  # Update with the correct port
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(4)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# Load YOLOv5 for stop sign detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Define target class
TARGET_CLASSES = ['stop sign']
last_detection_time = 0  # Tracks last detection time (seconds)

# Steering limits
STEERING_CENTER = 105
STEERING_LEFT = 80
STEERING_RIGHT = 130


def detect_stop_signs(frame):
    """Detects stop signs in the frame and signals Arduino to stop if detected."""
    global last_detection_time
    current_time = time.time()

    if current_time - last_detection_time < 5:
        return  

    resized_frame = cv2.resize(frame, (540, 260))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.xyxy[0]

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
                    time.sleep(2)  
                    arduino.write(b'GO\n')
                    print("Signal sent to Arduino: GO")
                    last_detection_time = time.time()
                except Exception as e:
                    print(f"Error sending signal to Arduino: {e}")
            break


def get_lane_center_offset(lanes_points):
    """Calculates lane center offset based on detected lanes."""
    if len(lanes_points) < 2:
        return None  # Not enough lanes detected
    
    left_lane = lanes_points[0]
    right_lane = lanes_points[1]
    
    if not left_lane and not right_lane:
        return None  # No valid lane data
    
    left_x = np.mean([pt[0] for pt in left_lane]) if left_lane else None
    right_x = np.mean([pt[0] for pt in right_lane]) if right_lane else None
    
    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) / 2
    elif left_x is not None:
        lane_center = left_x + 100  # Estimate right lane
    elif right_x is not None:
        lane_center = right_x - 100  # Estimate left lane
    else:
        return None  # No reliable lane center
    
    return lane_center


def adjust_steering(lanes_points):
    """Adjusts steering based on lane center offset."""
    lane_center = get_lane_center_offset(lanes_points)

    if lane_center is None:
        steering_angle = STEERING_CENTER  # Default to straight if no lanes are detected
        print("No lanes detected, going straight.")
    else:
        frame_center = 640  # Assuming a 1280x720 resolution
        offset = lane_center - frame_center  
        steering_angle = STEERING_CENTER - int(offset * 0.1)  

    steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))
    
    if arduino:
        try:
            arduino.write(f'{steering_angle}\n'.encode())
            print(f"Steering angle sent: {steering_angle}")
        except Exception as e:
            print(f"Error sending steering angle to Arduino: {e}")


# Main loop
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    lanes_points, output_img = lane_detector.detect_lanes(frame)  # Get lane points
    detect_stop_signs(frame)
    adjust_steering(lanes_points)

    cv2.imshow("Lane and Stop Sign Detection", output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
