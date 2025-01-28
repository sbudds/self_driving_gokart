import cv2
import torch
import serial
import time
import threading
import numpy as np

# Initialize the Arduino serial connection
SERIAL_PORT = '/dev/ttyACM0'  # Replace with your port
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow time for the serial connection to initialize
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the labels of interest (e.g., 'stop sign' label from your dataset)
TARGET_CLASSES = ['stop sign']

# Lane following-related functions
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < 0:  # Negative slope -> left line
                left_lines.append((slope, intercept))
            else:          # Positive slope -> right line
                right_lines.append((slope, intercept))

    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None

    return left_avg, right_avg

def calculate_steering(height, width, left_line):
    center_x = width // 2
    if left_line is not None:
        left_x = int((height - left_line[1]) / left_line[0])
    else:
        left_x = 0
    lane_center = left_x
    deviation = lane_center - center_x

    angle = 105 + (deviation / center_x) * 25  # Steering adjustment
    angle = max(80, min(130, angle))  # Clamp the angle
    return int(angle)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    roi_vertices = np.array([[
        (0, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width, height)
    ]], dtype=np.int32)
    cropped_edges = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)
    left_line, right_line = average_slope_intercept(lines)
    steering_angle = calculate_steering(height, width, left_line)
    return steering_angle

def lane_following():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        steering_angle = process_frame(frame)

        # Send the steering angle to Arduino
        if arduino:
            try:
                arduino.flush()
                arduino.write(f"{steering_angle}\n".encode())
                print(f"Steering Angle Sent: {steering_angle}")
            except Exception as e:
                print(f"Error sending steering angle to Arduino: {e}")

        time.sleep(0.1)  # Delay for processing time

    cap.release()

# Stop sign detection-related functions
def detect_stop_signs(frame):
    resized_frame = cv2.resize(frame, (640, 360))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    detections = results.xyxy[0]  # [x_min, y_min, x_max, y_max, confidence, class]
    for *box, confidence, cls in detections:
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES:
            print(f"Stop sign detected with confidence: {confidence:.2f}")

            # Send a signal to Arduino
            if arduino:
                try:
                    arduino.write(b'STOP\n')
                    print("Signal sent to Arduino: STOP")
                    time.sleep(4)  # Wait for 4 seconds (simulate stop)
                    arduino.write(b'GO\n')
                    print("Signal sent to Arduino: GO")
                except Exception as e:
                    print(f"Error sending signal to Arduino: {e}")
            break

def stop_sign_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        detect_stop_signs(frame)

    cap.release()

# Main function to run both tasks concurrently
def main():
    # Start the lane following thread
    lane_thread = threading.Thread(target=lane_following)
    lane_thread.daemon = True
    lane_thread.start()

    # Start the stop sign detection thread
    stop_sign_thread = threading.Thread(target=stop_sign_detection)
    stop_sign_thread.daemon = True
    stop_sign_thread.start()

    try:
        # Keep the main thread running while other threads are active
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program stopped by user.")

    # Cleanup
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()

