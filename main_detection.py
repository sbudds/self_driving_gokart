import cv2
import numpy as np
import Jetson.GPIO as GPIO
import time

# ------------------ Servo and GPIO Setup ------------------
SERVO_PIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(0)

# ------------------ Servo Limits ------------------
CENTER_ANGLE = 105
LEFT_ANGLE = 80
RIGHT_ANGLE = 130

# ------------------ Tuning Parameters ------------------
ANGLE_UPDATE_THRESHOLD = 4    # Only update servo if angle changes by > 4 degrees
STEERING_SENSITIVITY = 0.9    # Scale factor for deviation (1.0 = default, <1 less aggressive)
STEERING_BIAS = 0             # Additional offset in degrees (positive shifts right, negative shifts left)

# ------------------ Helper Functions ------------------
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
    histValues = np.sum(imgWarp[imgWarp.shape[0] // 2:, :], axis=0)  # Sum pixel values in lower half
    midpoint = len(histValues) // 2
    left_sum = np.sum(histValues[:midpoint])
    right_sum = np.sum(histValues[midpoint:])
    
    if left_sum + right_sum > 0:
        lane_center = np.argmax(histValues)  # Most prominent column
    else:
        lane_center = midpoint  # Default to center if no strong lane detected
    return lane_center

def getLaneCurve(img):
    """Process the image and determine lane deviation."""
    imgThres = thresholding(img)
    hT, wT = img.shape[:2]
    points = np.float32([(106, 111), (wT - 106, 111), (24, 223), (wT - 24, 223)])
    imgWarp = warpImg(imgThres, points, wT, hT)
    RoadCenter = getHistogram(imgWarp)
    imgCenter = wT // 2
    dist = RoadCenter - imgCenter
    return dist

def compute_steering_angle(dist):
    """Map lane deviation (dist) to a steering angle with tuning adjustments."""
    max_dist = 120  # Tuning parameter for sensitivity in original mapping
    k = (RIGHT_ANGLE - LEFT_ANGLE) / (2 * max_dist)
    # Base computed angle
    base_angle = CENTER_ANGLE + k * dist
    # Apply sensitivity and bias adjustments
    tuned_angle = CENTER_ANGLE + STEERING_SENSITIVITY * (base_angle - CENTER_ANGLE) + STEERING_BIAS
    # Clamp the angle within limits
    return max(min(tuned_angle, RIGHT_ANGLE), LEFT_ANGLE)

# ------------------ Main Loop ------------------
if __name__ == '__main__':
    # To use a video file instead of the camera,
    # uncomment the following line and comment out the camera line:
    # cap = cv2.VideoCapture("path/to/your_video.mp4")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use camera
    
    # Variable to hold the last updated angle (start at center)
    prev_angle = CENTER_ANGLE

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (480, 240))
        
        # To display the video footage, keep this line.
        # Comment it out to disable the window:
        cv2.imshow("Frame", img)
        
        # Compute lane deviation and desired steering angle
        dist = getLaneCurve(img)
        new_angle = compute_steering_angle(dist)

        # Update servo only if the change is significant
        if abs(new_angle - prev_angle) >= ANGLE_UPDATE_THRESHOLD:
            set_steering_angle(new_angle)
            prev_angle = new_angle

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
