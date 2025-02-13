import cv2
import numpy as np
import Jetson.GPIO as GPIO
import time

SERVO_PIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(0)

CENTER_ANGLE = 110
LEFT_ANGLE = 90
RIGHT_ANGLE = 130

ANGLE_UPDATE_THRESHOLD = 4    # Only update servo if angle changes by > 4 degrees
# Use separate sensitivities for left and right turns:
STEERING_SENSITIVITY_LEFT = 0.94    # For left turns (negative deviation)
STEERING_SENSITIVITY_RIGHT = 0.95   # For right turns (positive deviation)
STEERING_BIAS = 0           # Additional offset in degrees
SMOOTHING_FACTOR = 0.5      # For exponential smoothing (0: no update, 1: immediate update)

def angle_to_duty_cycle(angle):
    return (angle / 18.0) + 2.5

def set_steering_angle(angle):
    duty_cycle = angle_to_duty_cycle(angle)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)  

def thresholding(img):
    """
    Convert to grayscale, blur, apply adaptive thresholding, then smooth 
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # Smooth out small gaps/noise in lane lines
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def warpImg(img, points, wT, hT):
 
    src = np.float32(points)
    dst = np.float32([[0, 0], [wT, 0], [0, hT], [wT, hT]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    imgWarp = cv2.warpPerspective(img, matrix, (wT, hT))
    return imgWarp

def getLaneEdges(imgWarp):
    """
    In the warped image, consider only the bottom half.
    Then split it into left/right halves and compute the histogram for each.
    """
    h, w = imgWarp.shape
    bottom_half = imgWarp[h//2:, :]
    
    # Left half:
    left_half = bottom_half[:, :w//2]
    left_hist = np.sum(left_half, axis=0)
    left_edge_index = np.argmax(left_hist)
    
    # Right half:
    right_half = bottom_half[:, w//2:]
    right_hist = np.sum(right_half, axis=0)
    right_edge_index = np.argmax(right_hist)
    
    left_edge = left_edge_index               # relative to full image, in left half
    right_edge = right_edge_index + w//2       # offset for right half
    return left_edge, right_edge

def getLaneCurve(img):
    """
    Process the input image: threshold, warp, then use the lane edges from the left and right halves.
    Return the deviation (in pixels) of the lane center from the image center.
    """
    imgThres = thresholding(img)
    hT, wT = img.shape[:2]
    points = np.float32([(100, 120), (wT - 100, 120), (30, 240), (wT - 30, 240)])
    imgWarp = warpImg(imgThres, points, wT, hT)
    
    left_edge, right_edge = getLaneEdges(imgWarp)
    lane_center = (left_edge + right_edge) / 2.0
    img_center = wT / 2.0
    dist = lane_center - img_center
    return dist

def compute_steering_angle(dist, wT):
    """
    Map the lane deviation (in pixels) to a steering angle.
    Uses separate sensitivities for left and right turns.
    """
    max_dist = 120  # Tuning parameter: maximum expected deviation (in pixels)
    k = (RIGHT_ANGLE - LEFT_ANGLE) / (2 * max_dist)
    # Base angle based on deviation:
    base_angle = CENTER_ANGLE + k * dist
    
    # Apply different sensitivity based on turn direction:
    if dist > 0:
        sensitivity = STEERING_SENSITIVITY_RIGHT
    else:
        sensitivity = STEERING_SENSITIVITY_LEFT

    tuned_angle = CENTER_ANGLE + sensitivity * (base_angle - CENTER_ANGLE) + STEERING_BIAS
    return max(min(tuned_angle, RIGHT_ANGLE), LEFT_ANGLE)

if __name__ == '__main__':

    #cap = cv2.VideoCapture("/home/soumi/Videos/testvid (copy).mp4")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use camera
    
    smoothed_angle = CENTER_ANGLE

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (480, 272))
        
        cv2.imshow("Frame", img)
        
        dist = getLaneCurve(img)

        new_angle = compute_steering_angle(dist, img.shape[1])
        
        smoothed_angle = SMOOTHING_FACTOR * new_angle + (1 - SMOOTHING_FACTOR) * smoothed_angle

        if abs(smoothed_angle - CENTER_ANGLE) >= ANGLE_UPDATE_THRESHOLD:
            set_steering_angle(smoothed_angle)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
