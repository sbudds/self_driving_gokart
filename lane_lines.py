import cv2
import torch
import serial
import time
import numpy as np
import threading
import queue
import types
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

###########################
# GPU-BASED HELPER FUNCTIONS
###########################

def gpu_image_transform(frame):
    """
    Convert an OpenCV frame (BGR) to a normalized tensor on GPU.
    The lane detector expects an image of size (288, 800) in RGB.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(frame_rgb).float().cuda() / 255.0  # shape: (H, W, 3)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(288, 800),
                                                 mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.squeeze(0)

def gpu_process_output(output, cfg):
    """
    Process the lane detector model's output entirely on the GPU.
    Returns a tuple of (lanes_points, lanes_detected, lane_center) where:
      - lanes_points is a list of torch tensors (one per lane) with shape (N,2),
        each row being an (x,y) point.
      - lanes_detected is a boolean tensor (one value per lane).
      - lane_center is a scalar tensor containing the overall average x–coordinate
        of all detected lanes (used for computing the offset).
    """
    with torch.no_grad():
        processed_output = torch.flip(output[0], dims=[1])
        prob = torch.nn.functional.softmax(processed_output[:-1, :, :], dim=0)
        idx = (torch.arange(cfg.griding_num, device=processed_output.device).float() + 1.0).view(-1, 1, 1)
        loc = torch.sum(prob * idx, dim=0)
        argmax_output = torch.argmax(processed_output, dim=0)
        loc[argmax_output == cfg.griding_num] = 0
        
        col_sample = torch.linspace(0, 800 - 1, steps=cfg.griding_num, device=processed_output.device)
        col_sample_w = col_sample[1] - col_sample[0] if cfg.griding_num > 1 else torch.tensor(0.0, device=processed_output.device)
        x_coords = loc * (col_sample_w * cfg.img_w / 800.0) - 1.0
        
        valid_mask = loc > 0
        valid_counts = valid_mask.sum(dim=0).float()
        sum_x = (x_coords * valid_mask.float()).sum(dim=0)
        avg_x = torch.where(valid_counts > 0, sum_x / valid_counts, torch.zeros_like(sum_x))
        lanes_detected = valid_counts > 2
        
        if lanes_detected.any():
            lane_center = avg_x[lanes_detected].mean()
        else:
            lane_center = torch.tensor(cfg.img_w / 2.0, device=processed_output.device)
        
        # Prepare lane points for visualization.
        row_anchor = torch.tensor(cfg.row_anchor, device=processed_output.device, dtype=torch.float32)
        y_coords = cfg.img_h * (row_anchor.flip(0) / 288.0) - 1.0
        
        lanes_points = []
        for lane in range(loc.shape[1]):
            valid_indices = valid_mask[:, lane]
            if valid_indices.any():
                lane_x = x_coords[:, lane][valid_indices]
                lane_y = y_coords[valid_indices]
                lane_points = torch.stack([lane_x, lane_y], dim=1)
            else:
                lane_points = torch.empty((0, 2), device=processed_output.device)
            lanes_points.append(lane_points)
        
        return lanes_points, lanes_detected, lane_center

###########################
# PID CONTROLLER (Lightweight)
###########################

class PIDController:
    def __init__(self, kp, ki, kd, set_point=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.0:
            dt = 1e-3
        error = self.set_point - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        self.last_time = current_time
        return output

###########################
# Arduino Writer Thread
###########################

def arduino_writer(arduino, steering_queue):
    while True:
        angle = steering_queue.get()
        if angle is None:  # Sentinel value to shut down the thread.
            break
        # Drain any extra pending angles to only send the latest value.
        while not steering_queue.empty():
            try:
                angle = steering_queue.get_nowait()
            except queue.Empty:
                break
        try:
            arduino.write(f'{angle}\n'.encode())
            arduino.flush()  # Ensure the data is immediately sent out.
            time.sleep(0.05)  # Small delay (50ms) to give Arduino time to process.
        except Exception as e:
            print(f"[Arduino] Error sending steering angle: {e}")


###########################
# MAIN PROGRAM: LANE DETECTION & STEERING
###########################

def main():
    print("[Main] Waiting 7 seconds for system initialization...")
    time.sleep(7)

    model_path = "lanes/models/tusimple_18.pth"  # Update path as needed.
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    lane_detector.img_transform = gpu_image_transform
    lane_detector.process_output = gpu_process_output

    # Monkey-patch detect_lanes to store GPU results and draw lane lines.
    def detect_lanes_wrapper(self, frame):
        input_tensor = self.img_transform(frame).unsqueeze(0)
        output = self.model(input_tensor)
        lanes_points, lanes_detected, lane_center = self.process_output(output, self.cfg)
        self.lanes_points = lanes_points
        self.lanes_detected = lanes_detected
        self.lane_center = lane_center

        # Draw lane lines on the frame.
        # Define lane colors (BGR)
        lane_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for i, lane in enumerate(lanes_points):
            # Only draw if there are detected points.
            if lane.numel() > 0:
                # Transfer lane points from GPU to CPU for drawing.
                lane_cpu = lane.detach().cpu().numpy()
                points = []
                for point in lane_cpu:
                    # Convert the floating-point coordinates to integer pixel values.
                    x, y = int(point[0]), int(point[1])
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, lane_colors[i % len(lane_colors)], -1)
                if len(points) >= 2:
                    cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False,
                                  color=lane_colors[i % len(lane_colors)], thickness=2)
        return frame

    lane_detector.detect_lanes = types.MethodType(detect_lanes_wrapper, lane_detector)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Unable to access video source.")
        return
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    SERIAL_PORT = '/dev/ttyACM0'  # Update as needed.
    try:
        arduino = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        time.sleep(4)
        print(f"[Main] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"[Main] Error connecting to Arduino: {e}")
        arduino = None

    # Start Arduino writer thread.
    steering_queue = queue.Queue()
    if arduino:
        arduino_thread = threading.Thread(target=arduino_writer, args=(arduino, steering_queue), daemon=True)
        arduino_thread.start()

    pid = PIDController(0.1, 0.005, 0.05)
    STEERING_CENTER = 105
    STEERING_LEFT = 80
    STEERING_RIGHT = 130
    prev_steering_angle = STEERING_CENTER
    frame_width = lane_detector.cfg.img_w

    # Define how often (in frames) we update the steering command.
    PID_UPDATE_INTERVAL = 3

    print("[Main] Processing video input... (Press 'q' to quit)")
    frame_count = 0
    # Variables to store the latest steering values (for visualization)
    last_lane_center = frame_width / 2.0
    last_offset = 0.0
    last_correction = 0.0
    last_steering_angle = STEERING_CENTER

    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("[Main] Error: Unable to read frame.")
            break

        # Get the output frame with lane lines drawn.
        with torch.no_grad():
            output_img = lane_detector.detect_lanes(frame)

        # Only update the PID and send steering commands every PID_UPDATE_INTERVAL frames.
        if frame_count % PID_UPDATE_INTERVAL == 0:
            try:
                lane_center = lane_detector.lane_center
            except AttributeError:
                lane_center = torch.tensor(frame_width / 2.0, device='cuda')
            
            # Compute the horizontal offset (forcing a small GPU->CPU sync for one scalar)
            offset = (lane_center - torch.tensor(frame_width / 2.0, device=lane_center.device)).item()
            correction = pid.update(offset)
            new_steering_angle = STEERING_CENTER + correction

            STEERING_SMOOTHING = 0.2
            steering_angle = int((1 - STEERING_SMOOTHING) * new_steering_angle +
                                 STEERING_SMOOTHING * prev_steering_angle)
            prev_steering_angle = steering_angle
            steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))
            
            # Save these values for visualization.
            last_lane_center = lane_center.item()
            last_offset = offset
            last_correction = correction
            last_steering_angle = steering_angle

            if arduino:
                steering_queue.put(steering_angle)

        # Overlay the computed steering details on the output image.
        cv2.putText(output_img, f"Lane Center: {last_lane_center:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_img, f"Offset: {last_offset:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_img, f"PID Correction: {last_correction:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_img, f"Steering Angle: {last_steering_angle}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Lane Detection", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optionally: remove or adjust CUDA cache clearing.
        # if frame_count % 100 == 0:
        #     torch.cuda.empty_cache()

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        steering_queue.put(None)  # Signal the Arduino thread to shut down.
        arduino.close()

if __name__ == "__main__":
    main()
