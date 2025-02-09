import cv2
import torch
import numpy as np
import serial
import time

# Initialize Arduino Serial Connection
SERIAL_PORT = "/dev/ttyACM0"  # Update with your port
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
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

def process_frame_gpu(frame):
    """ Perform edge detection and Hough Transform using CUDA tensors. """
    # Convert frame to grayscale and move to GPU
    frame_tensor = torch.tensor(frame, dtype=torch.uint8, device="cuda")
    gray_tensor = 0.2989 * frame_tensor[:, :, 2] + 0.5870 * frame_tensor[:, :, 1] + 0.1140 * frame_tensor[:, :, 0]
    gray_tensor = gray_tensor.to(torch.uint8)

    # Apply Gaussian Blur (GPU)
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device="cuda") / 16
    gray_blur = torch.nn.functional.conv2d(gray_tensor.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()

    # Canny Edge Detection (GPU-based)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda")
    sobel_y = sobel_x.T
    grad_x = torch.nn.functional.conv2d(gray_blur.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    grad_y = torch.nn.functional.conv2d(gray_blur.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    edges = torch.sqrt(grad_x**2 + grad_y**2)
    edges = (edges > 50).to(torch.uint8) * 255  # Thresholding for edges

    # Hough Line Transform (Using GPU tensor operations)
    edge_coords = torch.nonzero(edges)  # Find edge pixels
    if edge_coords.shape[0] == 0:
        return None, None

    theta = torch.linspace(-np.pi / 2, np.pi / 2, 180, device="cuda")
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    rho_values = (edge_coords[:, 0, None] * cos_theta + edge_coords[:, 1, None] * sin_theta).to(torch.int32)
    rho_hist = torch.histc(rho_values.float(), bins=180, min=-edges.shape[0], max=edges.shape[1])
    
    line_indices = (rho_hist > 50).nonzero()[:, 0]  # Extract prominent lines
    detected_lines = [(rho_values[i].item(), theta[i].item()) for i in line_indices]

    return edges.cpu().numpy(), detected_lines  # Convert edges back for display

def get_lane_offset(lines, frame_width):
    """ Compute lane offset based on detected lines. """
    if not lines:
        return None

    left_lines = [rho for rho, theta in lines if rho < frame_width // 2]
    right_lines = [rho for rho, theta in lines if rho > frame_width // 2]

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
            arduino.write(f"{steering_angle}\n".encode())
            print(f"Steering angle sent: {steering_angle}")
        except Exception as e:
            print(f"Error sending steering angle: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    frame_width = frame.shape[1]

    edge_img, lines = process_frame_gpu(frame)
    offset = get_lane_offset(lines, frame_width)
    adjust_steering(offset)

    # Display edges on CPU
    if edge_img is not None:
        cv2.imshow("Lane Detection (GPU)", edge_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
