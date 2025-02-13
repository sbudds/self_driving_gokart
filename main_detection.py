import cv2
import numpy as np
import utils
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

# Convert angle to duty cycle
def angle_to_duty_cycle(angle):
    return (angle / 18.0) + 2.5

def set_steering_angle(angle):
    duty_cycle = angle_to_duty_cycle(angle)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)  # Stop sending PWM signal after setting angle

def getLaneCurve(img, display=2):
    imgThres = utils.thresholding(img)
    hT, wT = img.shape[:2]
    points = np.float32([(106, 111), (480-106, 111), (24, 223), (480-24, 223)])
    imgWarp = utils.warpImg(imgThres, points, wT, hT)
    RoadCenter, _ = utils.getHistogram(imgWarp, display=False, minPer=0.5, region=4)
    imgCenter = wT // 2
    dist = RoadCenter - imgCenter
    return dist

def compute_steering_angle(dist):
    max_dist = 120  # Tuning parameter for sensitivity
    k = (RIGHT_ANGLE - LEFT_ANGLE) / (2 * max_dist)
    angle = CENTER_ANGLE + k * dist
    return max(min(angle, RIGHT_ANGLE), LEFT_ANGLE)

if __name__ == '__main__':
    cap = cv2.VideoCapture("video.mp4")
    while cap.isOpened():  
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (480, 240))
        dist = getLaneCurve(img, display=2)
        steering_angle = compute_steering_angle(dist)
        set_steering_angle(steering_angle)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
