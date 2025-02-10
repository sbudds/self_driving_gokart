import cv2
import numpy as np
import serial
import time

# Set up serial communication with the Arduino
def setup_serial():
    # Open the serial port where the Arduino is connected
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)  # Wait for the serial connection to be established
    return arduino

# Send the steering angle to the Arduino
def control_steering(arduino, angle):
    # Map the steering angle from (-1 to 1) to (80 to 130 degrees)
    mapped_angle = np.clip(105 + angle * 25, 80, 130)  # 105 is center, 80 is max left, 130 is max right
    steering_command = str(int(mapped_angle))
    arduino.write(steering_command.encode())  # Send the command to the Arduino
    print(f"Sent steering angle: {steering_command}")

# Region of interest (ROI) where lane lines are expected
def region_of_interest(img):
    height, width = img.shape
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.5), int(height * 0.5))
    ]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Convert frame to GPU
def process_frame_to_gpu(frame):
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    return gpu_frame

# CUDA edge detection (Canny)
def canny_edge_detection(gpu_frame):
    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale on GPU
    gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)     # Gaussian blur on GPU
    gpu_edges = cv2.cuda.Canny(gpu_blurred, 50, 150)              # Canny edge detection on GPU
    return gpu_edges

# CUDA Hough Transform for line detection
def detect_lane_lines_on_gpu(gpu_edges):
    lines = cv2.cuda.HoughLinesP(gpu_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    return lines

# Calculate the steering angle with proportional control
def calculate_steering(left_lines, right_lines, frame):
    if not left_lines or not right_lines:
        return 0  # No lines detected, keep steering straight
    
    # Calculate the average position of the lane lines
    left_x = np.mean([x1 for x1, y1, x2, y2 in left_lines])
    right_x = np.mean([x1 for x1, y1, x2, y2 in right_lines])

    center_x = (left_x + right_x) / 2
    frame_center = frame.shape[1] / 2

    error = center_x - frame_center

    # Proportional control (steering adjusts based on error)
    steering_angle = -error * 0.01  # Tweak this multiplier for smoother control
    return steering_angle

# Main loop
def main():
    cap = cv2.VideoCapture(0)  # Use 0 for the first camera device, replace if needed
    arduino = setup_serial()   # Set up serial communication with the Arduino

    time.sleep(5)  # Wait for 5 seconds for initialization (e.g., camera and serial setup)

    frame_skip = 2  # Skip every other frame to reduce CPU load
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Skip frames to reduce CPU usage
        if frame_counter % frame_skip == 0:
            # Resize the frame to reduce resolution and GPU load
            frame = cv2.resize(frame, (640, 480))  # Reduce resolution (optional)

            # Process frame using GPU
            gpu_frame = process_frame_to_gpu(frame)

            # Perform edge detection on GPU
            gpu_edges = canny_edge_detection(gpu_frame)

            # Detect lane lines using Hough Transform on GPU
            lines = detect_lane_lines_on_gpu(gpu_edges)

            if lines is None:
                continue

            left_lines = []
            right_lines = []
            height, width = frame.shape[:2]
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                if slope > 0:  # Right lane
                    right_lines.append((x1, y1, x2, y2))
                else:  # Left lane
                    left_lines.append((x1, y1, x2, y2))

            # Calculate steering angle if lanes are detected
            steering_angle = calculate_steering(left_lines, right_lines, frame)
            control_steering(arduino, steering_angle)

            # Draw the detected lane lines on the frame (using CPU for drawing only)
            for line in left_lines:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
            for line in right_lines:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

            cv2.imshow("Lane Detection", frame)

        frame_counter += 1

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    arduino.close()  # Close the serial connection when done

if __name__ == "__main__":
    main()
