import cv2
import numpy as np
import torch
import serial
import time

# Initialize Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'  # Update with your port
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}: {e}")
    arduino = None

# Steering Constants
STEERING_CENTER = 105
STEERING_LEFT = 80
STEERING_RIGHT = 130

# Open GStreamer Pipeline for Direct GPU Capture
cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'q' to quit)")

def process_frame_gpu(frame):
    """ Uses CUDA for Edge Detection and Line Detection."""
    frame_gpu = torch.tensor(frame, device='cuda', dtype=torch.uint8)
    
    # Convert to Grayscale on GPU
    gray_gpu = 0.2989 * frame_gpu[..., 2] + 0.5870 * frame_gpu[..., 1] + 0.1140 * frame_gpu[..., 0]
    gray_gpu = gray_gpu.to(torch.uint8)
    
    # Apply Canny Edge Detection on GPU
    edges_gpu = cv2.cuda.createCannyEdgeDetector(50, 150)
    edges = edges_gpu.detect(gray_gpu)
    
    # Apply Hough Transform on GPU
    lines_gpu = cv2.cuda.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    return edges, lines_gpu

def get_lane_center_offset(lines, frame_width):
    """ Calculate lane center offset from detected lines. """
    if lines is None:
        return None
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, _, x2, _ = line[0]
        x_avg = (x1 + x2) // 2
        if x_avg < frame_width // 2:
            left_lines.append(x_avg)
        else:
            right_lines.append(x_avg)
    
    if not left_lines or not right_lines:
        return None
    
    left_x = np.mean(left_lines)
    right_x = np.mean(right_lines)
    lane_center = (left_x + right_x) / 2
    return lane_center - (frame_width / 2)

def adjust_steering(offset):
    """ Adjust steering based on lane offset. """
    if offset is None:
        steering_angle = STEERING_CENTER
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
    edges, lines = process_frame_gpu(frame)
    offset = get_lane_center_offset(lines, frame_width)
    adjust_steering(offset)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
