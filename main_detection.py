import cv2
import torch
import serial
import time

# Initialize the Arduino serial connection
# Replace 'dev/tty/ACM0' with the appropriate port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
SERIAL_PORT = 'dev/tty/ACM0'
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow time for the serial connection to initialize
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# Load the YOLOv5 model
# Replace 'yolov5s' with the path to your custom-trained YOLOv5 model if necessary
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the labels of interest (e.g., 'stop sign' label from your dataset)
TARGET_CLASSES = ['stop sign']  # Update this based on your model's class names

# Function to process the frame and detect stop signs
def detect_stop_signs(frame):
    # Resize the frame to a smaller resolution for faster processing
    resized_frame = cv2.resize(frame, (640, 360))

    # Convert resized frame to RGB as YOLOv5 model expects RGB input
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(rgb_frame)

    # Parse results
    detections = results.xyxy[0]  # [x_min, y_min, x_max, y_max, confidence, class]
    for *box, confidence, cls in detections:
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES:
            print(f"Stop sign detected with confidence: {confidence:.2f}")

            # Send a signal to the Arduino
            if arduino:
                try:
                    arduino.write(b'STOP\n')  # Replace 'STOP' with the desired signal
                    print("Signal sent to Arduino: STOP")

                    # Wait for 4 seconds (simulate waiting at the stop sign)
                    time.sleep(4)

                    # Send signal to resume movement (replace 'GO' with the signal your system uses)
                    arduino.write(b'GO\n')
                    print("Signal sent to Arduino: GO")

                except Exception as e:
                    print(f"Error sending signal to Arduino: {e}")

            # Break after detecting the first stop sign to avoid redundant processing
            break

# Start video capture (from webcam or video file)
cap = cv2.VideoCapture(0)  # Use '0' for webcam or replace with video file path

if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'Ctrl+C' to stop)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Detect stop signs in the frame
        detect_stop_signs(frame)

        # No video feed is displayed; processing runs in the background
except KeyboardInterrupt:
    print("Video processing stopped by user.")

# Release resources
cap.release()

if arduino:
    arduino.close()
