import cv2
import torch
import serial
import time
import numpy as np

# Initialize CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'  # Update with the correct port
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

def preprocess_frame(frame):
    """ Convert frame to grayscale and send to GPU as a tensor. """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_tensor = torch.tensor(frame_gray, dtype=torch.float32, device=device) / 255.0
    return frame_tensor

def detect_edges(frame_tensor):
    """ Apply Sobel edge detection using CUDA tensors. """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    edge_x = torch.nn.functional.conv2d(frame_tensor, sobel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(frame_tensor, sobel_y, padding=1)

    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze()
    edges = (edges > 0.2).float()  # Thresholding
    return edges

def refine_lanes(edges):
    """ Apply morphological operations to clean up lane lines. """
    kernel = torch.ones((1, 5), dtype=torch.float32, device=device)  # 1x5 kernel
    edges = torch.nn.functional.conv2d(edges.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=(0, 2))
    return edges.squeeze()

def get_lane_center_offset(edges, frame_width):
    """ Calculate the lane center offset using lane pixel distribution. """
    edges_np = edges.cpu().numpy()
    y_indices, x_indices = np.where(edges_np > 0)

    if len(x_indices) == 0:
        return None  # No lanes detected

    lane_center = np.mean(x_indices)
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
    
    frame_tensor = preprocess_frame(frame)
    edges = detect_edges(frame_tensor)
    edges = refine_lanes(edges)
    offset = get_lane_center_offset(edges, frame_width)
    adjust_steering(offset)

    # Convert edges back to an image for display
    edges_np = (edges.cpu().numpy() * 255).astype(np.uint8)
    cv2.imshow("Lane Detection", edges_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
