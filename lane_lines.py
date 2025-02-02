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
cap.set(3, 640)  # Set width
cap.set(4, 360)  # Set height

frame_skip = 2  # Skip every 2 frames for faster processing
previous_angle = 105  # Start with center position (105 degrees)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def calculate_steering_angle(frame_width, contours, prev_angle):
    center_angle = 105
    angle_range = 25
    min_angle = center_angle - angle_range
    max_angle = center_angle + angle_range

    if len(contours) == 0:
        return prev_angle

    left_contours = [contour for contour in contours if np.mean(contour[:, 0, 0]) < frame_width / 2]
    if len(left_contours) == 0:
        return prev_angle

    largest_contour = max(left_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]

    steering_angle = prev_angle * 0.7 + angle * 0.3
    steering_angle = np.clip(steering_angle, min_angle, max_angle)
    return int(steering_angle)

def process_frame(frame):
    frame_gpu = torch.from_numpy(frame).to(device, dtype=torch.float32) / 255.0
    gray_gpu = torch.mean(frame_gpu, dim=2, keepdim=True)
    binary_gpu = (gray_gpu > 0.4).float()
    
    binary_cpu = (binary_gpu * 255).byte().cpu().numpy()
    contours, _ = cv2.findContours(binary_cpu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, binary_cpu

def send_steering_angle(angle):
    arduino.write(f"{angle}\n".encode())
    print(f"Steering Angle: {angle}")

def main():
    global previous_angle
    frame_count = 0
    torch.cuda.empty_cache()

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
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
