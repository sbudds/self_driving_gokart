import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox

cfg_path = os.path.abspath("yolo-cfg/yolov3.cfg")
weights_path = os.path.abspath("yolo-cfg/yolov3.weights")
names_path = os.path.abspath("yolo-cfg/coco.names")

net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()

try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

lower_red1 = np.array([170, 204, 166])
upper_red1 = np.array([180, 255, 255])
lower_red2 = np.array([0, 191, 153])
upper_red2 = np.array([10, 255, 255])
lower_yellow = np.array([17, 204, 153])
upper_yellow = np.array([32, 255, 255])
lower_green = np.array([42, 104, 178])
upper_green = np.array([70, 255, 255])

current_hsv = None

def adjust_hsv_range(hsv_value, color):
    global lower_red1, upper_red1, lower_red2, upper_red2
    global lower_yellow, upper_yellow, lower_green, upper_green
    
    h, s, v = hsv_value
    h = int(h)
    s = int(s)
    v = int(v)

    if color == 'red':
        if h < 10: 
            lower_red2[0] = min(lower_red2[0], h)
            upper_red2[0] = max(upper_red2[0], h)
        else:  
            lower_red1[0] = min(lower_red1[0], h)
            upper_red1[0] = max(upper_red1[0], h)

       
        lower_yellow[0] = max(lower_yellow[0], h + 1) if h >= lower_yellow[0] else lower_yellow[0]
        lower_green[0] = max(lower_green[0], h + 1) if h >= lower_green[0] else lower_green[0]

    elif color == 'yellow':
        lower_yellow[0] = min(lower_yellow[0], h)
        upper_yellow[0] = max(upper_yellow[0], h)

        if h < 10:
            upper_red2[0] = min(upper_red2[0], h - 1)
        else:
            upper_red1[0] = min(upper_red1[0], h - 1)
        lower_green[0] = max(lower_green[0], h + 1) if h >= lower_green[0] else lower_green[0]

    elif color == 'green':
        lower_green[0] = min(lower_green[0], h)
        upper_green[0] = max(upper_green[0], h)

        
        if h < 10:
            upper_red2[0] = min(upper_red2[0], h - 1)
        else:
            upper_red1[0] = min(upper_red1[0], h - 1)
        upper_yellow[0] = min(upper_yellow[0], h - 1)

    print(f"Adjusted HSV ranges for {color}:")
    print(f"Red: lower1={lower_red1}, upper1={upper_red1}, lower2={lower_red2}, upper2={upper_red2}")
    print(f"Yellow: lower={lower_yellow}, upper={upper_yellow}")
    print(f"Green: lower={lower_green}, upper={upper_green}")

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    global current_hsv
    current_hsv = cv2.mean(hsv)[:3]  

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    if cv2.countNonZero(mask_red):
        return "red"
    elif cv2.countNonZero(mask_yellow):
        return "yellow"
    elif cv2.countNonZero(mask_green):
        return "green"
    else:
        return "unknown"

def create_gui():
    def on_red():
        if current_hsv is not None:
            adjust_hsv_range(current_hsv, 'red')
            messagebox.showinfo("Info", "Red HSV range adjusted")

    def on_yellow():
        if current_hsv is not None:
            adjust_hsv_range(current_hsv, 'yellow')
            messagebox.showinfo("Info", "Yellow HSV range adjusted")

    def on_green():
        if current_hsv is not None:
            adjust_hsv_range(current_hsv, 'green')
            messagebox.showinfo("Info", "Green HSV range adjusted")

    window = tk.Tk()
    window.title("Adjust Traffic Light Color")

    red_button = tk.Button(window, text="Red", command=on_red, height=2, width=10)
    red_button.pack(pady=10)

    yellow_button = tk.Button(window, text="Yellow", command=on_yellow, height=2, width=10)
    yellow_button.pack(pady=10)

    green_button = tk.Button(window, text="Green", command=on_green, height=2, width=10)
    green_button.pack(pady=10)

    window.mainloop()

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                roi = frame[y:y+h, x:x+w]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                if label == "traffic light":
                    detected_color = detect_color(roi)
                    if detected_color == "red":
                        color = (0, 0, 255)
                    elif detected_color == "yellow":
                        color = (0, 255, 255)
                    elif detected_color == "green":
                        color = (0, 255, 0)
                
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Color: {detected_color}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                elif label == "stop sign":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Stop Sign", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import threading
gui_thread = threading.Thread(target=create_gui)
gui_thread.start()

main()
