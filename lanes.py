import cv2
import numpy as np
import torch
import utlis
import serial
import time

# Initialize Arduino Serial Connection
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
    time.sleep(7)
except Exception as e:
    print(f"Error: Could not connect to Arduino: {e}")
    arduino = None

# Steering Angles
STEERING_CENTER = 105
STEERING_LEFT = 80
STEERING_RIGHT = 130
last_steering_angle = None  # Track last sent steering angle

# Open Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Processing video input... (Press 'q' to quit)")

curveList = []
avgVal = 10

def process_frame_gpu(frame):
    """ Process frame for lane curve using GPU. """
    frame_tensor = torch.tensor(frame, dtype=torch.uint8, device="cuda")
    imgThres = utlis.thresholding(frame_tensor.cpu().numpy())  # Run thresholding on CPU but process on GPU

    hT, wT, c = frame.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(frame, points)

    # Histogram and curve processing using GPU
    middlePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    curve = curve / 100
    if curve > 1:
        curve = 1
    if curve < -1:
        curve = -1

    return curve, imgWarpPoints, imgWarp, imgHist

def adjust_steering(offset):
    """ Adjust steering and only send if it changes. """
    global last_steering_angle
    if offset is None:
        steering_angle = STEERING_CENTER
    else:
        steering_angle = STEERING_CENTER - int(offset * 0.05)
        steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

    if steering_angle != last_steering_angle:
        if arduino:
            try:
                arduino.write(f"{steering_angle}\n".encode())
                print(f"Steering angle sent: {steering_angle}")
            except Exception as e:
                print(f"Error sending steering angle: {e}")
        last_steering_angle = steering_angle

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Resize frame and send to GPU processing
    frame = cv2.resize(frame, (480, 240))
    
    # Get lane curve and process the frame
    curve, imgWarpPoints, imgWarp, imgHist = process_frame_gpu(frame)
    print(f"curve = {curve}")

    # Adjust steering based on detected lane curve
    offset = np.mean(np.where(imgHist > 0)[1]) - (frame.shape[1] / 2) if np.any(imgHist > 0) else None
    adjust_steering(offset)

    # Show full pipeline with all steps
    imgStacked = utlis.stackImages(0.7, ([frame, imgWarpPoints, imgWarp], [imgHist, imgHist, imgHist]))
    cv2.imshow("Lane Detection (GPU)", imgStacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if arduino:
    arduino.close()
