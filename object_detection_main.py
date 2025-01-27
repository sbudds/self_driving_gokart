import cv2
import torch

# Load the YOLOv5 model
# Replace 'yolov5s' with the path to your custom-trained YOLOv5 model if necessary
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the labels of interest (e.g., 'stop sign' label from your dataset)
TARGET_CLASSES = ['stop sign']  # Update this based on your model's class names

# Function to process the frame and detect stop signs
def detect_stop_signs(frame):
    # Resize the frame to a smaller resolution for faster processing
    resized_frame = cv2.resize(frame, (640, 360))

    # Convert resized frame to RGB as YOLOv5 model expects RGB input
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(rgb_frame)

    # Parse results
    detections = results.xyxy[0]  # [x_min, y_min, x_max, y_max, confidence, class]
    for *box, confidence, cls in detections:
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES:
            x_min, y_min, x_max, y_max = map(int, box)
            # Scale box coordinates back to the original frame size
            scale_x = frame.shape[1] / resized_frame.shape[1]
            scale_y = frame.shape[0] / resized_frame.shape[0]
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)

            # Draw a rectangle around the detected stop sign
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Start video capture (from webcam or video file)
cap = cv2.VideoCapture(0)  # Use '0' for webcam or replace with video file path

if not cap.isOpened():
    print("Error: Unable to access video source.")
    exit()

print("Press 'q' to quit the video stream.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Detect stop signs in the frame
    processed_frame = detect_stop_signs(frame)

    # Display the frame
    cv2.imshow('Stop Sign Detection', processed_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
