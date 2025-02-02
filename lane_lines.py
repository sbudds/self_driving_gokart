import torch
import cv2
import numpy as np
import serial
import time

# Setup for Arduino communication
arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

# Setup Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 360)  # Height

frame_skip = 2  
previous_angle = 105  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def calculate_steering_angle(frame_width, contours, prev_angle):
    center_angle = 105
    angle_range = 25
    min_angle = center_angle - angle_range
    max_angle = center_angle + angle_range

    if len(contours) == 0:
        return prev_angle

    left_contours = [c for c in contours if np.mean(c[:, 0, 0]) < frame_width / 2]
    if len(left_contours) == 0:
        return prev_angle

    largest_contour = max(left_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]

    steering_angle = prev_angle * 0.7 + angle * 0.3
    return int(np.clip(steering_angle, min_angle, max_angle))

def process_frame(frame):
    # Upload frame to GPU
    frame_gpu = cv2.cuda_GpuMat()
    frame_gpu.upload(frame)

    # Convert to grayscale
    gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_gpu = cv2.cuda.GaussianBlur(gray_gpu, (5, 5), 0)

    # Canny edge detection
    edges_gpu = cv2.cuda.Canny(blurred_gpu, 50, 150)

    # Download result to CPU for contour detection
    edges_cpu = edges_gpu.download()
    
    contours, _ = cv2.findContours(edges_cpu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, edges_cpu

def send_steering_angle(angle):
    arduino.write(f"{angle}\n".encode())
    print(f"Steering Angle: {angle}")

def main():
    global previous_angle
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        contours, edges = process_frame(frame)
        steering_angle = calculate_steering_angle(frame.shape[1], contours, previous_angle)
        previous_angle = steering_angle
        send_steering_angle(steering_angle)
        
        cv2.imshow("Edges", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        torch.cuda.empty_cache()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
