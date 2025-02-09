import cv2
import numpy as np
import vpi
import serial
import time

# Initialize Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'  # Update with your port
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(4)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# Steering Angles
STEERING_CENTER = 105
STEERING_LEFT = 80
STEERING_RIGHT = 130

# Open Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'q' to quit)")

def process_frame_vpi(frame):
    """ Uses NVIDIA VPI for Canny Edge Detection and Hough Transform. """
    height, width = frame.shape[:2]

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the NumPy array to a VPI image in CPU memory
    with vpi.Image.from_cpu(gray, vpi.Format.U8) as vpi_gray:
        # Use VPI for Canny Edge Detection
        edges = vpi_gray.canny(low_threshold=50, high_threshold=150)

        # Convert back to NumPy array
        edge_img = edges.cpu()

    # Use VPI for Hough Line Detection
    with vpi.Image.from_cpu(edge_img, vpi.Format.U8) as vpi_edges:
        hough_lines = vpi_edges.hough_lines(rho=1, theta=np.pi / 180, threshold=50)

        # Convert lines to NumPy array
        lines_np = hough_lines.cpu()

    return edge_img, lines_np

def get_lane_center_offset(lines, frame_width):
    """ Calculate lane center offset from detected lines. """
    if lines is None or len(lines) == 0:
        return None

    left_lines = []
    right_lines = []
    
    for line in lines:
        rho, theta = line[0], line[1]
        x1 = int(rho * np.cos(theta) + 1000 * (-np.sin(theta)))
        y1 = int(rho * np.sin(theta) + 1000 * (np.cos(theta)))
        x2 = int(rho * np.cos(theta) - 1000 * (-np.sin(theta)))
        y2 = int(rho * np.sin(theta) - 1000 * (np.cos(theta)))

        # Classify the lines
        if x1 < frame_width // 2 and x2 < frame_width // 2:
            left_lines.append((x1, x2))
        elif x1 > frame_width // 2 and x2 > frame_width // 2:
            right_lines.append((x1, x2))

    if not left_lines or not right_lines:
        return None

    left_x = np.mean([pt[0] for pt in left_lines])
    right_x = np.mean([pt[0] for pt in right_lines])
    lane_center = (left_x + right_x) / 2
    return lane_center - (frame_width / 2)

def adjust_steering(offset):
    """ Adjust steering based on lane offset. """
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

    edge_img, lines = process_frame_vpi(frame)
    offset = get_lane_center_offset(lines, frame_width)
    adjust_steering(offset)

    # Display the detected edges
    cv2.imshow("Lane Detection (VPI)", edge_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
