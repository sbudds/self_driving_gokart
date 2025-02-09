import cv2
import numpy as np
import vpi
import serial
import time

# Initialize Arduino Serial Connection
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
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
    """ Process the frame using GPU-accelerated VPI operations """
    frame_height, frame_width = frame.shape[:2]

    with vpi.Backend.CUDA:
        with vpi.Image.from_host(frame, vpi.Format.RGB8) as vpi_frame:
            # Convert to grayscale
            vpi_gray = vpi_frame.convert(vpi.Format.U8, backend=vpi.Backend.CUDA)

            # Apply Gaussian Blur
            vpi_blurred = vpi_gray.gaussian_filter(kernel_size=5, backend=vpi.Backend.CUDA)

            # Sobel Edge Detection (faster than Canny)
            vpi_edges = vpi_blurred.sobel_filter(backend=vpi.Backend.CUDA)

            # Convert edges back to NumPy (small data transfer)
            edges = vpi_edges.cpu()

    # Apply threshold to detect lane lines
    _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # Define ROI mask (only process bottom half of the image)
    mask = np.zeros_like(binary_edges)
    roi_vertices = np.array([[(0, frame_height), (frame_width, frame_height), 
                              (frame_width//2, frame_height//2)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(binary_edges, mask)

    # Fit lines using a fast CUDA-based regression (avoid HoughLinesP bottleneck)
    left_line, right_line = fit_lines_cuda(masked_edges)

    return masked_edges, left_line, right_line


def fit_lines_cuda(edge_img):
    """ Custom CUDA-based line fitting to avoid HoughLinesP bottlenecks """
    # Convert edge image to points
    points = np.column_stack(np.where(edge_img > 0))

    if len(points) < 50:  # No significant lane detected
        return None, None

    # Split left and right lanes
    mid_x = edge_img.shape[1] // 2
    left_points = points[points[:, 1] < mid_x]
    right_points = points[points[:, 1] >= mid_x]

    # Fit lines using NumPy for now (can be replaced with CUDA kernel)
    left_line = np.polyfit(left_points[:, 0], left_points[:, 1], 1) if len(left_points) > 10 else None
    right_line = np.polyfit(right_points[:, 0], right_points[:, 1], 1) if len(right_points) > 10 else None

    return left_line, right_line


def adjust_steering(left_line, right_line, frame_width):
    """ Adjust steering based on detected lane lines """
    if left_line is None and right_line is None:
        steering_angle = STEERING_CENTER
    elif left_line is None:
        steering_angle = STEERING_RIGHT
    elif right_line is None:
        steering_angle = STEERING_LEFT
    else:
        # Find the midpoint between detected lanes
        lane_center = (np.polyval(left_line, frame_width//2) + np.polyval(right_line, frame_width//2)) / 2
        steering_angle = STEERING_CENTER - int((lane_center - frame_width//2) * 0.05)
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

    # Use CUDA-optimized lane detection
    edge_img, left_line, right_line = process_frame_vpi(frame)

    # Adjust the car's steering
    adjust_steering(left_line, right_line, frame_width)

    # Display results
    cv2.imshow("Lane Detection (CUDA Optimized)", edge_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
