import cv2
import time
import lane_lines
import stop_signs

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Unable to access video source.")
        return

    smoothed_angle = lane_lines.CENTER_ANGLE

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        frame = cv2.resize(frame, (480, 272))

        lane_deviation = lane_lines.getLaneCurve(frame)
        new_angle = lane_lines.compute_steering_angle(lane_deviation, frame.shape[1])

        smoothed_angle = (lane_lines.SMOOTHING_FACTOR * new_angle +
                          (1 - lane_lines.SMOOTHING_FACTOR) * smoothed_angle)
        if abs(smoothed_angle - lane_lines.CENTER_ANGLE) >= lane_lines.ANGLE_UPDATE_THRESHOLD:
            lane_lines.set_steering_angle(smoothed_angle)

        stop_signs.detect_stop_signs(frame)

        cv2.imshow("Combined Video Stream", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    lane_lines.servo.stop()
    lane_lines.GPIO.cleanup()

    if stop_signs.arduino is not None:
        stop_signs.arduino.close()

if __name__ == "__main__":
    main()
