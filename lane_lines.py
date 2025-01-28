import cv2
import numpy as np
import serial
import time

# Initialize the serial connection to Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Replace with your Arduino's serial port
time.sleep(2)  # Wait for the connection to initialize

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < 0:  # Negative slope -> left line
                left_lines.append((slope, intercept))
            else:          # Positive slope -> right line
                right_lines.append((slope, intercept))

    # Average slope and intercept for left and right lanes
    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None

    return left_avg, right_avg

def calculate_steering(height, width, left_line):
    center_x = width // 2

    # Calculate x position of the left line at the bottom of the frame
    if left_line is not None:
        left_x = int((height - left_line[1]) / left_line[0])
    else:
        left_x = 0

    # Find the lane center (assuming a single line to the left of the car)
    lane_center = left_x

    # Deviation from the frame center
    deviation = lane_center - center_x

    # Calculate steering angle
    if deviation < 0:
        angle = 105 + (deviation / center_x) * 25  # Steer left
    else:
        angle = 105 + (deviation / center_x) * 25  # Steer right

    # Clamp angle between 80 and 130 degrees
    angle = max(80, min(130, angle))

    return int(angle)

def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest
    height, width = edges.shape
    roi_vertices = np.array([[
        (0, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width, height)
    ]], dtype=np.int32)

    cropped_edges = region_of_interest(edges, roi_vertices)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)

    # Calculate average lane lines
    left_line, right_line = average_slope_intercept(lines)

    # Calculate steering angle (for the left line)
    steering_angle = calculate_steering(height, width, left_line)

    return steering_angle

def main():
    # Open the camera feed
    cap = cv2.VideoCapture(0)  # Use the appropriate index for your camera

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame from camera.")
            break

        # Process the frame and get the steering angle
        steering_angle = process_frame(frame)

        # Send steering angle to Arduino (with newline)
        arduino.flush()  # Ensure the serial buffer is cleared
        arduino.write(f"{steering_angle}\n".encode())  # Ensure the newline character is sent

        # Print the steering angle to the terminal
        print(f"Steering Angle: {steering_angle}")

        time.sleep(0.1)  # Optional: Small delay before sending the next angle

if __name__ == "__main__":
    main()
