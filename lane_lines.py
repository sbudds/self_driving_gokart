import cv2
import torch
import serial
import time
import numpy as np
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
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to torch tensor and move to GPU (still on CPU until .cuda() call)
    img_tensor = torch.from_numpy(frame_rgb).float().cuda() / 255.0  # shape: (H, W, 3)
    # Change shape to (1, 3, H, W)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    # Resize on GPU to (1, 3, 288, 800)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(288, 800),
                                                 mode='bilinear', align_corners=False)
    # Normalize using standard mean and std
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.squeeze(0)  # shape: (3, 288, 800)

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
        # (1) Process the raw output on GPU
        processed_output = torch.flip(output[0], dims=[1])
        prob = torch.nn.functional.softmax(processed_output[:-1, :, :], dim=0)
        idx = (torch.arange(cfg.griding_num, device=processed_output.device).float() + 1.0).view(-1, 1, 1)
        loc = torch.sum(prob * idx, dim=0)
        argmax_output = torch.argmax(processed_output, dim=0)
        loc[argmax_output == cfg.griding_num] = 0
        
        # (2) Compute x–coordinates for each lane point on GPU
        col_sample = torch.linspace(0, 800 - 1, steps=cfg.griding_num, device=processed_output.device)
        col_sample_w = col_sample[1] - col_sample[0] if cfg.griding_num > 1 else torch.tensor(0.0, device=processed_output.device)
        x_coords = loc * (col_sample_w * cfg.img_w / 800.0) - 1.0  # shape: (num_rows, max_lanes)
        
        # (3) Determine which points are valid (loc > 0)
        valid_mask = loc > 0
        valid_counts = valid_mask.sum(dim=0).float()  # per-lane count (shape: (max_lanes,))
        sum_x = (x_coords * valid_mask.float()).sum(dim=0)
        avg_x = torch.where(valid_counts > 0, sum_x / valid_counts, torch.zeros_like(sum_x))
        
        # A lane is "detected" if it has more than 2 valid points
        lanes_detected = valid_counts > 2
        
        # (4) Compute overall lane center: the mean of avg_x for all detected lanes.
        if lanes_detected.any():
            lane_center = avg_x[lanes_detected].mean()
        else:
            lane_center = torch.tensor(cfg.img_w / 2.0, device=processed_output.device)
        
        # (5) (Optional) Create lanes_points for visualization.
        # For each row, we want to compute a y–coordinate.
        # The original code uses: y = int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane - 1 - point_num] / 288)) - 1
        # We vectorize this by first converting row_anchor to a tensor and then flipping it.
        row_anchor = torch.tensor(cfg.row_anchor, device=processed_output.device, dtype=torch.float32)
        y_coords = cfg.img_h * (row_anchor.flip(0) / 288.0) - 1.0  # shape: (num_rows,)
        
        lanes_points = []
        for lane in range(loc.shape[1]):
            valid_indices = valid_mask[:, lane]
            if valid_indices.any():
                lane_x = x_coords[:, lane][valid_indices]
                lane_y = y_coords[valid_indices]
                lane_points = torch.stack([lane_x, lane_y], dim=1)  # shape: (N,2)
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
        self.set_point = set_point  # Target: lane center exactly at frame center
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
# MAIN PROGRAM: LANE DETECTION & STEERING
###########################

def main():
    # Wait for system and GPU initialization.
    print("[Main] Waiting 7 seconds for system initialization...")
    time.sleep(7)

    # Initialize the lane detection model.
    model_path = "lanes/models/tusimple_18.pth"  # Update path if needed.
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

    # Override the image transformation and output processing with our GPU-based versions.
    lane_detector.img_transform = gpu_image_transform
    lane_detector.process_output = gpu_process_output

    # --- Monkey-patch detect_lanes to store lane_center, lanes_points, and lanes_detected ---
    def detect_lanes_wrapper(self, frame):
        # Transform the frame on GPU
        input_tensor = self.img_transform(frame)
        input_tensor = input_tensor.unsqueeze(0)  # add batch dimension if needed
        # Run the model (assumed to be on GPU)
        output = self.model(input_tensor)
        # Process the output on GPU
        lanes_points, lanes_detected, lane_center = self.process_output(output, self.cfg)
        # Store results in the detector object for later use
        self.lanes_points = lanes_points
        self.lanes_detected = lanes_detected
        self.lane_center = lane_center
        # For visualization, you might choose to draw the lanes.
        # (Here we simply return the original frame as a placeholder.)
        return frame

    lane_detector.detect_lanes = types.MethodType(detect_lanes_wrapper, lane_detector)
    # ----------------------------------------------------------------------------------

    # Set up video capture.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Unable to access video source.")
        return
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    # Initialize Arduino connection for steering commands.
    SERIAL_PORT = '/dev/ttyACM0'  # Update port if necessary.
    try:
        arduino = serial.Serial(SERIAL_PORT, 9600, timeout=1)
        time.sleep(4)
        print(f"[Main] Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"[Main] Error connecting to Arduino: {e}")
        arduino = None

    # Set up PID controller for steering.
    pid = PIDController(0.1, 0.005, 0.05)
    STEERING_CENTER = 105  # Neutral steering angle.
    STEERING_LEFT = 80     # Maximum left.
    STEERING_RIGHT = 130   # Maximum right.
    prev_steering_angle = STEERING_CENTER
    frame_width = lane_detector.cfg.img_w  # Expected frame width (e.g., 1280).

    print("[Main] Processing video input... (Press 'q' to quit)")
    frame_count = 0
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("[Main] Error: Unable to read frame.")
            break

        # Run lane detection on the current frame.
        with torch.no_grad():
            output_img = lane_detector.detect_lanes(frame)

        # Retrieve the lane_center computed on GPU.
        try:
            # lane_center is a torch tensor on GPU.
            lane_center = lane_detector.lane_center
        except AttributeError:
            lane_center = torch.tensor(frame_width / 2.0, device='cuda')

        # Compute the horizontal offset on GPU then transfer a single scalar to CPU.
        offset_tensor = lane_center - torch.tensor(frame_width / 2.0, device=lane_center.device)
        offset = offset_tensor.item()  # float value

        # Update PID controller and compute new steering angle.
        correction = pid.update(offset)
        new_steering_angle = STEERING_CENTER + correction

        # Apply smoothing to reduce jitter.
        STEERING_SMOOTHING = 0.2
        steering_angle = int((1 - STEERING_SMOOTHING) * new_steering_angle +
                             STEERING_SMOOTHING * prev_steering_angle)
        prev_steering_angle = steering_angle
        steering_angle = max(STEERING_LEFT, min(STEERING_RIGHT, steering_angle))

        # Send the steering command to the Arduino.
        if arduino:
            try:
                arduino.write(f'{steering_angle}\n'.encode())
            except Exception as e:
                print(f"[Main] Error sending steering angle: {e}")

        # Display the lane detection output.
        cv2.imshow("Lane Detection", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optionally clear the CUDA cache every 100 frames.
        if frame_count % 100 == 0:
            torch.cuda.empty_cache()

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()
