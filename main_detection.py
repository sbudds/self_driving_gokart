import cv2
import torch
import serial
import time

from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# Model paths and settings
model_path = "lanes/models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True 
weights_only = True

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

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

# Load the YOLOv5 model with CUDA support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Define target class
TARGET_CLASSES = ['stop sign']
last_detection_time = 0  # Tracks last detection time (seconds)

def detect_stop_signs(frame):
    global last_detection_time
    current_time = time.time()
    
    if current_time - last_detection_time < 5:
        return frame  # Skip detection if within 5 seconds of last one
    
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
    
    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Lane & Stop Sign Detection", cv2.WINDOW_NORMAL)

print("Processing video input... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break
    
    # Detect lanes
    lanes_frame = lane_detector.detect_lanes(frame)
    
    # Detect stop signs
    output_frame = detect_stop_signs(lanes_frame)
    
    # Display results
    cv2.imshow("Lane & Stop Sign Detection", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
