#!/usr/bin/env python3
import math
import cv2
import numpy as np
import time
import serial

# Function to calculate slope (this will run on the CPU since it is not a GPU-accelerated operation)
def slope(vx1, vx2, vy1, vy2):
    m = float(vy2 - vy1) / float(vx2 - x1)  # Slope equation
    theta1 = math.atan(m)  # Calculate the slope angle
    return theta1 * (180 / np.pi)  # Return the calculated angle in degrees

# Set up the video capture
cap = cv2.VideoCapture(0)

# Set up serial communication with Arduino
def setup_serial():
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)  # Wait for the serial connection to establish
    return arduino

# Send the steering angle to Arduino
def control_steering(arduino, angle):
    mapped_angle = 105 + angle * 25  # The multiplier is adjusted for smoother control
    mapped_angle = np.clip(mapped_angle, 80, 130)  # Limit the steering angle between 80 and 130
    steering_command = str(int(mapped_angle))  # Convert the angle to a string
    arduino.write(steering_command.encode())  # Send the steering command to Arduino
    print(f"Sent steering angle: {steering_command}")

# Initialize variables
a = b = c = 1
arduino = setup_serial()  # Set up serial communication

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.resize(img, (600, 600))

    # Upload the frame to GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    # Convert the frame to grayscale on the GPU
    gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

    # Equalize histogram on GPU (if needed)
    gpu_eq = cv2.cuda.equalizeHist(gpu_gray)

    # Apply Gaussian blur on GPU
    gpu_blurred = cv2.cuda.createGaussianFilter(gpu_eq, -1, (5, 5), 0).apply(gpu_eq)

    # Threshold the image (GPU)
    _, gpu_thresh = cv2.cuda.threshold(gpu_blurred, 240, 255, cv2.THRESH_BINARY)

    # Download the result back to CPU for contour finding (GPU contour is not available)
    thresh = gpu_thresh.download()

    # Find contours (only CPU operation)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours (CPU-based)
    cv2.drawContours(thresh, contours, -1, (255, 0, 0), 3)

    # Process the image to detect lane lines
    drawing = np.zeros(img.shape, np.uint8)

    # Apply Hough Transform on GPU to detect lines
    gpu_lines = cv2.cuda.HoughLinesP(gpu_thresh, 1, np.pi / 180, 25, minLineLength=10, maxLineGap=40)
    lines = gpu_lines.download()  # Download lines to CPU

    left_x = []
    right_x = []
    l = r = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if round(x2 - x1) != 0:
                arctan = slope(x1, x2, y1, y2)

                if 250 < y1 < 600 and 250 < y2 < 600:
                    # Left lane lines
                    if -80 <= round(arctan) <= -30:
                        r += 1
                        l = 0
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                        left_x.append((x1 + x2) / 2)  # Store x positions of left lane lines

                    # Right lane lines
                    if 30 <= round(arctan) <= 80:
                        l += 1
                        r = 0
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                        right_x.append((x1 + x2) / 2)  # Store x positions of right lane lines

    # Calculate the steering angle with proportional control
    if left_x and right_x:
        left_avg_x = np.mean(left_x)
        right_avg_x = np.mean(right_x)
        center_x = (left_avg_x + right_avg_x) / 2  # Calculate the center of the lanes
        frame_center = img.shape[1] / 2  # The center of the frame

        # Calculate the error from the center
        error = center_x - frame_center
        steering_angle = -error * 0.01  # Adjust multiplier for smoother control

        # Send the steering angle to Arduino
        control_steering(arduino, steering_angle)
    else:
        print('No lane detected, keeping straight...')
        control_steering(arduino, 0)  # Keep the car straight if no lanes are detected

    # Display the results
    cv2.imshow('Thresholded Image', thresh)
    cv2.imshow('Detected Lines', img)

    # Decision making based on detected lines
    if l >= 10 and a == 1:
        print('left')
        a = 0
        b = 1
        c = 1
    elif r >= 10 and b == 1:
        print('right')
        a = 1
        b = 0
        c = 1
    elif l < 10 and r < 10 and c == 1:
        print('straight')
        a = 1
        b = 1
        c = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()  # Close the serial connection when done
