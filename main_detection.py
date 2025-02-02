import torch
import cv2
import serial
import time

# Initialize the Arduino serial connection
SERIAL_PORT = '/dev/ttyACM0'  # Update with the correct port
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# Load YOLOv5 model with CUDA support (for lane detection as well)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Define target class (For lane detection, we need to adjust this to lane-related classes)
TARGET_CLASSES = ['lane']  # This will depend on your dataset if you fine-tune YOLOv5

last_detection_time = 0  # Tracks last detection time (seconds)

def detect_lanes(frame):
    global last_detection_time

    current_time = time.time()  # Get current time

    # Ignore detection if it's within 5 seconds of the last one
    if current_time - last_detection_time < 5:
        return  

    resized_frame = cv2.resize(frame, (540, 260))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)  # Model inference directly on the GPU
    detections = results.xyxy[0]

    # Process detection results
    for *box, confidence, cls in detections:
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Lane detected with confidence: {confidence:.2f}")
            
            # Send signal to Arduino (steering control)
            if arduino:
                try:
                    arduino.write(b'STOP\n')
                    print("Signal sent to Arduino: STOP")
                    time.sleep(2)  # Wait before sending GO command
                    arduino.write(b'GO\n')
                    print("Signal sent to Arduino: GO")
                    last_detection_time = time.time()  # Update last detection time
                except Exception as e:
                    print(f"Error sending signal to Arduino: {e}")
            break

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Perform lane detection
    detect_lanes(frame)
    cv2.imshow("Lane Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if arduino:
    arduino.close()
