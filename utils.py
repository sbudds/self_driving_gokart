import cv2
import numpy as np

# Thresholding function to create a binary mask based on HSV thresholds
def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

# Initialize trackbars for adjusting threshold values
def initializeTrackbars(initialVals):
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("Hue Min", "Trackbars", initialVals[0], 179, nothing)
    cv2.createTrackbar("Hue Max", "Trackbars", initialVals[1], 179, nothing)
    cv2.createTrackbar("Sat Min", "Trackbars", initialVals[2], 255, nothing)
    cv2.createTrackbar("Sat Max", "Trackbars", initialVals[3], 255, nothing)

def nothing(x):
    pass

# Get trackbar values (for adjusting thresholds)
def valTrackbars():
    hMin = cv2.getTrackbarPos("Hue Min", "Trackbars")
    hMax = cv2.getTrackbarPos("Hue Max", "Trackbars")
    sMin = cv2.getTrackbarPos("Sat Min", "Trackbars")
    sMax = cv2.getTrackbarPos("Sat Max", "Trackbars")
    vMin = 100
    vMax = 255
    return np.array([(0, 0, vMin), (hMax, sMax, vMax)], dtype=np.uint8)

# Warp the image based on perspective points (Region of Interest)
def warpImg(img, points, w, h, inv=False):
    matrix = cv2.getPerspectiveTransform(np.float32(points), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    if inv:
        matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])), np.float32(points)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp

# Function to draw points on the image
def drawPoints(img, points):
    for point in points:
        cv2.circle(img, tuple(point), 10, (0, 0, 255), cv2.FILLED)
    return img

# Calculate histogram and return the middle point and histogram image
def getHistogram(img, display=False, minPer=0.1, region=1):
    histogram = np.sum(img[img.shape[0]//region:, :], axis=0)
    curve = histogram / np.max(histogram)
    middlePoint = np.int(np.argmax(curve))
    if display:
        histImg = np.zeros_like(img)
        for x in range(0, len(curve)):
            cv2.line(histImg, (x, img.shape[0]), (x, img.shape[0] - np.int(curve[x] * img.shape[0])), (0, 255, 0), 2)
        return middlePoint, histImg
    return middlePoint, None

# Stack multiple images together for display
def stackImages(scale, imgArray):
    # Stacking images in a grid for display
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    imgWidth = imgArray[0][0].shape[1]
    imgHeight = imgArray[0][0].shape[0]

    # Creating an empty array to stack images
    hor = [imgArray[i][0] for i in range(rows)]
    ver = cv2.hconcat(hor)
    return ver


