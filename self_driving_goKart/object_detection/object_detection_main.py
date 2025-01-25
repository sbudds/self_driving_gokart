import cv2
import numpy as np
from ultralytics import YOLO  # Install this via `pip install ultralytics`

# Load the fine-tuned YOLOv8 model (replace with your trained model's path)
model = YOLO("yolov8n_traffic_light.pt")  # Ensure you have this file in your working directory

# Define HSV ranges for traffic light colors
HSV_RANGES = {
    "red": [(0, 50, 50), (10, 255, 255)],    # Red lower range
    "yellow": [(15, 100, 100), (35, 255, 255)],
    "green": [(40, 50, 50), (80, 255, 255)],
}

# Video feed setup
cap = cv2.VideoCapture(0)  # Replace with your webcam or video file

def detect_traffic_light_color(roi):
    """Detect traffic light color within the ROI using HSV masks."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in HSV_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 0:  # If significant color detected
            return color
    return "unknown"

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Run YOLO detection
        results = model(frame)

        for box in results[0].boxes:  # Iterate through detected objects
            conf = box.conf.item()
            if conf < 0.5:  # Confidence threshold
                continue

            # Extract bounding box and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results[0].names[int(box.cls[0])]

            # Process only traffic lights
            if label == "traffic light":
                roi = frame[y1:y2, x1:x2]  # Region of Interest (ROI)
                detected_color = detect_traffic_light_color(roi)

                # Draw bounding box and label
                color = (0, 255, 0) if detected_color == "green" else (0, 255, 255) if detected_color == "yellow" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{detected_color.capitalize()} Light", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the video feed
        cv2.imshow("Traffic Light Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
