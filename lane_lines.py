import torch
import cv2
import numpy as np
import serial
import time

# Setup for Arduino communication (Ensure correct port is used)
arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

# Setup Video Capture
cap = cv2.VideoCapture(0)  # Adjust the source as needed

# Reduce resolution for faster processing
cap.set(3, 640)  # Set width
cap.set(4, 360)  # Set height

# Skip frames for performance improvement
frame_skip = 2  # Skip every 2 frames for faster processing

# Smooth the steering by keeping track of previous angles
previous_angle = 105  # Start with center position (105 degrees)

# Device setup (make sure your device is CUDA-capable)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_steering_angle(frame_height, contours, prev_angle):
    center_angle = 105
    angle_range = 25
    min_angle = center_angle - angle_range
    max_angle = center_angle + angle_range

    if len(contours) == 0:
        return prev_angle  # No lines detected, return previous angle for smoothing

    # Filter contours for the left side (e.g., left lane line detection)
    left_contours = [contour for contour in contours if np.mean(contour[:, 0, 0]) < frame_height / 2]

    if len(left_contours) == 0:
        return prev_angle  # No left lane detected, return previous angle

    # Find the angle of the largest contour on the left side
    largest_contour = max(left_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]  # The rotation angle of the bounding box

    # Apply smooth steering by adjusting to a weighted average of current and previous angles
    steering_angle = prev_angle * 0.7 + angle * 0.3  # 70% weight on previous, 30% on current
    steering_angle = np.clip(steering_angle, min_angle, max_angle)

    return int(steering_angle)

def process_frame(frame):
    # Upload frame to GPU (ensure it's on GPU)
    gpu_frame = torch.tensor(frame).to(device).float()  # Move the frame to GPU
    gpu_frame /= 255.0  # Normalize the frame if needed

    # Convert to grayscale (make sure the tensor stays on GPU)
    gray_tensor = gpu_frame.mean(dim=2, keepdim=True)  # Average over the color channels

    # Apply threshold for edge detection (as an alternative to Canny)
    thresholded_tensor = gray_tensor > 0.4  # Use a simple threshold for binary image creation

    # Explicitly move to CPU for further processing and convert back to numpy
    binary_frame_cpu = thresholded_tensor.cpu().numpy().astype(np.uint8) * 255

    # Find contours on the edges (CPU-based)
    contours, _ = cv2.findContours(binary_frame_cpu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Free GPU memory periodically to avoid memory fragmentation
    torch.cuda.empty_cache()

    return contours, binary_frame_cpu

def send_steering_angle(angle):
    # Send steering angle to Arduino
    arduino.write(f"{angle}\n".encode())
    print(f"Steering Angle: {angle}")  # Print the steering angle for debugging

def main():
    global previous_angle
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for performance boost
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Process the frame to detect lanes
        contours, edges = process_frame(frame)

        # Calculate steering angle based on lane detection
        steering_angle = calculate_steering_angle(frame.shape[0], contours, previous_angle)

        # Update the previous angle for smooth control
        previous_angle = steering_angle

        # Send the calculated angle to the Arduino
        send_steering_angle(steering_angle)

        # Display the frame with detected contours (for debugging purposes)
        cv2.imshow("Edges", edges)

        # Debugging: Monitor GPU usage and device status
        print(f"GPU Device: {torch.cuda.current_device()}, Memory Allocated: {torch.cuda.memory_allocated()}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
