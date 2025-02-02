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
device = torch.device('cuda')  # Force GPU usage
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
    with torch.no_grad():  # No gradients needed for inference
        # Convert frame to tensor and move to GPU
        frame_gpu = torch.tensor(frame, device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0  

        # Convert to grayscale using PyTorch (keep data on GPU)
        gray_gpu = 0.299 * frame_gpu[0] + 0.587 * frame_gpu[1] + 0.114 * frame_gpu[2]  

        # Apply edge detection manually (avoiding OpenCV)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        gray_gpu = gray_gpu.unsqueeze(0).unsqueeze(0)  # Convert to (N, C, H, W) for conv2d
        
        edges_x = torch.nn.functional.conv2d(gray_gpu, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(gray_gpu, sobel_y, padding=1)
        
        edges_gpu = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        edges_gpu = (edges_gpu > 0.4).float()  # Thresholding

        # Move processed frame back to CPU for contour detection
        edges_cpu = (edges_gpu * 255).byte().squeeze().cpu().numpy()
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
