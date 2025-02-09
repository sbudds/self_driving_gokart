import cv2
import numpy as np
import torch
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
    print(f"Error: Could not connect to Arduino: {e}")
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

def tensor_canny_hough(frame):
    """CUDA-accelerated edge detection and Hough transform using PyTorch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert frame to grayscale and normalize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
    tensor_img = torch.tensor(gray, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Sobel filter for edge detection (Canny alternative)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    edge_x = torch.nn.functional.conv2d(tensor_img, sobel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(tensor_img, sobel_y, padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2).squeeze().cpu().numpy()

    # Apply thresholding
    edges = (edges > 0.3).astype(np.uint8) * 255

    # Perform Hough Transform on GPU
    edge_tensor = torch.tensor(edges, dtype=torch.float32, device=device)
    return edges, detect_lane_lines(edge_tensor, frame.shape[1])

def detect_lane_lines(edge_tensor, width):
    """Extract left and right lane lines from edge tensor using GPU calculations"""
    y_indices, x_indices = torch.where(edge_tensor > 0)
    if len(x_indices) == 0:
        return None

    left_x = x_indices[x_indices < width // 2].float().mean() if torch.any(x_indices < width // 2) else None
    right_x = x_indices[x_indices >= width // 2].float().mean() if torch.any(x_indices >= width // 2) else None

    if left_x is None or right_x is None:
        return None

    lane_center = (left_x + right_x) / 2
    return lane_center.item() - (width / 2)

def adjust_steering(offset):
    """Adjust steering based on lane offset"""
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
            print(f"Error sending steering angle: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    edges, offset = tensor_canny_hough(frame)
    adjust_steering(offset)

    cv2.imshow("Lane Detection (CUDA)", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
