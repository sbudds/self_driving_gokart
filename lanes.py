import cv2
import numpy as np
import serial
import time

# Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'  # Update with the correct port
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(4)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# Steering Angle Limits
STEERING_CENTER = 105
STEERING_LEFT = 80
STEERING_RIGHT = 130

# Open Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'q' to quit)")

def preprocess_frame(frame):
    """ Convert frame to grayscale, apply Gaussian blur and Canny edge detection. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_lane_lines(edges):
    """ Detect lane lines using Hough Transform. """
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)
    return lines

def get_lane_center_offset(lines, frame_width):
    """ Calculate the offset of lane center from frame center. """
    if lines is None or len(lines) == 0:
        return None  # No lines detected

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero

        if slope < -0.3:  # Left lane
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0.3:  # Right lane
            right_lines.append((x1, y1, x2, y2))

    if not left_lines and not right_lines:
        return None  # No valid lane lines detected

    left_x = np.mean([x1 + x2 for x1, y1, x2, y2 in left_lines]) / 2 if left_lines else None
    right_x = np.mean([x1 + x2 for x1, y1, x2, y2 in right_lines]) / 2 if right_lines else None

    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) / 2
    elif left_x is not None:
        lane_center = left_x + 100
    elif right_x is not None:
        lane_center = right_x - 100
    else:
        return None

    return lane_center - (frame_width / 2)

def adjust_steering(offset):
    """ Adjust steering based on lane center offset. """
    if offset is None:
        steering_angle = STEERING_CENTER  # Go straight if no lanes detected
    else:
        steering_angle = STEERING_CENTER - int(offset * 0.05)
        steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

    if arduino:
        try:
            arduino.write(f'{steering_angle}\n'.encode())
            print(f"Steering angle sent: {steering_angle}")
        except Exception as e:
            print(f"Error sending steering angle to Arduino: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    frame_width = frame.shape[1]
    
    edges = preprocess_frame(frame)
    lines = detect_lane_lines(edges)
    offset = get_lane_center_offset(lines, frame_width)
    adjust_steering(offset)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Lane Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
