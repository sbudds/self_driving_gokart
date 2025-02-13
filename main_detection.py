import cv2
import numpy as np
import Jetson.GPIO as GPIO
import time

# Set up GPIO for servo control
SERVO_PIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(0)

# Servo limits
CENTER_ANGLE = 115
LEFT_ANGLE = 90
RIGHT_ANGLE = 140

# Smoothing and compensation parameters
smoothing_factor = 0.2       # 0: no update, 1: instant update; lower means smoother
compensation_factor = 0.1    # 10% reduction in deviation from center

# Convert angle to duty cycle
def angle_to_duty_cycle(angle):
    return (angle / 18.0) + 2.5

def set_steering_angle(angle):
    duty_cycle = angle_to_duty_cycle(angle)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)  # Stop sending PWM signal after setting angle

def thresholding(img):
    """Apply grayscale, Gaussian blur, and adaptive thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def warpImg(img, points, wT, hT):
    """Apply perspective transformation to warp the image."""
    src = np.float32(points)
    dst = np.float32([[0, 0], [wT, 0], [0, hT], [wT, hT]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    imgWarp = cv2.warpPerspective(img, matrix, (wT, hT))
    return imgWarp

def getHistogram(imgWarp):
    """Compute the histogram to find the lane center."""
    histValues = np.sum(imgWarp[imgWarp.shape[0]//2:, :], axis=0)  # Sum pixel values in lower half
    midpoint = len(histValues) // 2
    left_sum = np.sum(histValues[:midpoint])
    right_sum = np.sum(histValues[midpoint:])
    
    # Determine lane center shift
    if left_sum + right_sum > 0:
        lane_center = np.argmax(histValues)  # Most prominent column
    else:
        lane_center = midpoint  # Default to center if no strong lane detected
    
    return lane_center

def getLaneCurve(img):
    """Process the image and determine lane deviation."""
    imgThres = thresholding(img)
    hT, wT = img.shape[:2]
    points = np.float32([(106, 111), (wT-106, 111), (24, 223), (wT-24, 223)])
    imgWarp = warpImg(imgThres, points, wT, hT)
    RoadCenter = getHistogram(imgWarp)
    imgCenter = wT // 2
    dist = RoadCenter - imgCenter
    return dist

def compute_steering_angle(dist):
    """Map distance deviation to steering angle."""
    max_dist = 120  # Tuning parameter for sensitivity
    k = (RIGHT_ANGLE - LEFT_ANGLE) / (2 * max_dist)
    angle = CENTER_ANGLE + k * dist
    return max(min(angle, RIGHT_ANGLE), LEFT_ANGLE)

if __name__ == '__main__':
    # To use a video file instead of the camera,
    # uncomment the following line and comment out the camera line:
    cap = cv2.VideoCapture("/home/soumi/Downloads/IMG_0955.mp4")
    
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use camera

    prev_angle = CENTER_ANGLE  # Initialize with the center angle
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (480, 240))
        
        # Display the current frame (optional)
        cv2.imshow("Frame", img)
        
        # Calculate lane deviation and raw steering angle
        dist = getLaneCurve(img)
        raw_angle = compute_steering_angle(dist)
        
        # Apply compensation to delay turning (reduce deviation from center)
        corrected_angle = CENTER_ANGLE + (raw_angle - CENTER_ANGLE) * (1 - compensation_factor)
        
        # Smooth the steering angle using a weighted average with the previous angle
        smoothed_angle = prev_angle * (1 - smoothing_factor) + corrected_angle * smoothing_factor
        prev_angle = smoothed_angle  # Update for next iteration
        
        # Set the servo to the smoothed steering angle
        set_steering_angle(smoothed_angle)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
