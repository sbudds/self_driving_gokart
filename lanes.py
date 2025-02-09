import cv2
import numpy as np
import serial
import time

# Initialize Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'
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

# CUDA Edge Detection Function
def process_frame_cuda(frame):
    """ Uses OpenCV's CUDA functions for edge detection and Hough Line Transform. """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Upload image to GPU
    gpu_gray = cv2.cuda_GpuMat()
    gpu_gray.upload(gray)

    # Canny edge detection using CUDA
    gpu_edges = cv2.cuda.createCannyEdgeDetector(50, 150)
    gpu_edges = gpu_edges.detect(gpu_gray)

    # Download the result from GPU
    edge_img = gpu_edges.download()

    # Hough Line Transform using CUDA
    lines = cv2.cuda.createHoughLineDetector(1, np.pi / 180, 50)
    gpu_lines = lines.detect(gpu_edges)

    return edge_img, gpu_lines

def get_lane_center_offset(lines, frame_width):
    """ Calculate lane center offset from detected lines. """
    if lines is None or len(lines) == 0:
        return None

    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x = (x1 + x2) // 2  # Find mid-point of line
        if x < frame_width // 2:
            left_lines.append(x)
        else:
            right_lines.append(x)

    if not left_lines or not right_lines:
        return None

    left_x = np.mean(left_lines)
    right_x = np.mean(right_lines)
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

    edge_img, lines = process_frame_cuda(frame)
    offset = get_lane_center_offset(lines, frame_width)
    adjust_steering(offset)

    # Display the detected edges
    cv2.imshow("Lane Detection (CUDA)", edge_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
