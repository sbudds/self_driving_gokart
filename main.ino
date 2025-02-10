#!/usr/bin/env python3
import math
import cv2
import numpy as np
import time
import serial

def slope(x1, x2, y1, y2):
    """
    Calculate the slope angle (in degrees) of a line segment given by (x1, y1) to (x2, y2).
    Returns 0 if the line is vertical.
    """
    if (x2 - x1) == 0:
        return 0
    m = float(y2 - y1) / float(x2 - x1)
    theta = math.atan(m)
    return theta * (180 / np.pi)

def setup_serial():
    """Initialize serial communication with the Arduino."""
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(7)  # Allow time for Arduino to fully initialize
    return arduino

def control_steering(arduino, command):
    """
    Send the steering command to the Arduino.
    command: a string representing the servo angle ("80", "105", or "130").
    """
    arduino.write(command.encode())  # Send the command to Arduino
    print(f"Sent steering command: {command}")

# Set up video capture and Arduino connection.
cap = cv2.VideoCapture(0)
arduino = setup_serial()

# State flags (optional) to avoid sending duplicate commands repeatedly.
a = b = c = 1

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Resize for consistent processing.
    img = cv2.resize(img, (600, 600))

    # --- GPU Processing ---
    # Upload frame to GPU.
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    # Convert to grayscale on the GPU.
    gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

    # Equalize histogram on the GPU.
    gpu_eq = cv2.cuda.equalizeHist(gpu_gray)

    # Apply Gaussian blur on the GPU (using a 5x5 kernel).
    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
    gpu_blur = gaussian_filter.apply(gpu_eq)

    # Apply binary threshold on the GPU.
    # Use a threshold of 240 so that only the brightest (white) pixels remain.
    retval, gpu_thresh = cv2.cuda.threshold(gpu_blur, 240, 255, cv2.THRESH_BINARY)

    # Download the thresholded image back to CPU.
    thresh = gpu_thresh.download()
    # ---------------------

    # (Optional) Draw contours for debugging.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh, contours, -1, (255, 0, 0), 3)

    # Use GPU-based Hough Segment Detector to detect lines.
    # Parameters: rho=1, theta=pi/180, threshold=25, minLineLength=10, maxLineGap=40.
    detector = cv2.cuda.createHoughSegmentDetector(1, np.pi/180, 25, 10, 40)
    gpu_lines = detector.detect(gpu_thresh)
    lines = gpu_lines.download() if gpu_lines is not None else None

    # Initialize counters for left and right lane lines.
    l = 0
    r = 0

    if lines is not None:
        for line in lines:
            # Each line is represented as [x1, y1, x2, y2].
            for x1, y1, x2, y2 in line:
                if round(x2 - x1) == 0:
                    continue  # Skip vertical lines to avoid division by zero.
                ang = slope(x1, x2, y1, y2)
                # Process only lines in a region of interest (y between 250 and 600).
                if 250 < y1 < 600 and 250 < y2 < 600:
                    # If the slope angle is between -80 and -30, count it as a right lane.
                    if -80 <= round(ang) <= -30:
                        r += 1
                        # Draw line for debugging.
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                    # If the slope angle is between 30 and 80, count it as a left lane.
                    if 30 <= round(ang) <= 80:
                        l += 1
                        # Draw line for debugging.
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    # Decision-making based on the counters.
    # (State flags a, b, c help prevent resending the same command repeatedly.)
    if l >= 10 and a == 1:
        print("Detected left lane markers; steering left.")
        control_steering(arduino, "80")    # Send left turn command.
        a = 0; b = 1; c = 1
    elif r >= 10 and b == 1:
        print("Detected right lane markers; steering right.")
        control_steering(arduino, "130")   # Send right turn command.
        a = 1; b = 0; c = 1
    elif l < 10 and r < 10 and c == 1:
        print("No clear lane markers; steering straight.")
        control_steering(arduino, "105")   # Send straight command.
        a = 1; b = 1; c = 0

    # Display images for debugging.
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Detected Lines", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
