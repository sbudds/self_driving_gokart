import cv2
import torch
import numpy as np
import serial
import time

# Initialize Arduino Serial Connection
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
    time.sleep(7)
except Exception as e:
    print(f"Error: Could not connect to Arduino: {e}")
    arduino = None

# Steering Angles
STEERING_CENTER = 105
STEERING_LEFT = 80
STEERING_RIGHT = 130
last_steering_angle = None  # Track last sent steering angle

# Open Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'q' to quit)")

def process_frame_gpu(frame):
    """ Perform edge detection and Hough Transform using CUDA tensors. """
    frame_tensor = torch.tensor(frame, dtype=torch.uint8, device="cuda")
    gray_tensor = (0.2989 * frame_tensor[:, :, 2] + 
                   0.5870 * frame_tensor[:, :, 1] + 
                   0.1140 * frame_tensor[:, :, 0]).to(torch.uint8)

    # Apply Gaussian Blur (GPU)
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device="cuda") / 16
    gray_blur = torch.nn.functional.conv2d(gray_tensor.unsqueeze(0).unsqueeze(0), 
                                           kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()

    # Canny Edge Detection (GPU)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda")
    sobel_y = sobel_x.T
    grad_x = torch.nn.functional.conv2d(gray_blur.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    grad_y = torch.nn.functional.conv2d(gray_blur.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    edges = torch.sqrt(grad_x**2 + grad_y**2)
    edges = (edges > 50).to(torch.uint8) * 255  # Thresholding for edges

    return edges.cpu().numpy()  # Convert to NumPy for display

def adjust_steering(offset):
    """ Adjust steering and only send if it changes. """
    global last_steering_angle
    if offset is None:
        steering_angle = STEERING_CENTER
    else:
        steering_angle = STEERING_CENTER - int(offset * 0.05)
        steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

    if steering_angle != last_steering_angle:
        if arduino:
            try:
                arduino.write(f"{steering_angle}\n".encode())
                print(f"Steering angle sent: {steering_angle}")
            except Exception as e:
                print(f"Error sending steering angle: {e}")
        last_steering_angle = steering_angle

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    frame_width = frame.shape[1]

    edge_img = process_frame_gpu(frame)

    # Display video output
    cv2.imshow("Lane Detection (GPU)", edge_img)

    # Only check for steering if there are edges
    if edge_img is not None:
        offset = np.mean(np.where(edge_img > 0)[1]) - (frame_width / 2) if np.any(edge_img > 0) else None
        adjust_steering(offset)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
