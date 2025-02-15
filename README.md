# Self-Driving RC Car with YOLOv5, OpenCV, and NVIDIA Jetson

This project is a self-driving RC car that uses an NVIDIA Jetson for image processing and direct steering control via a servo motor, along with an Arduino for motor control. The system is capable of detecting lane lines and stop signs using a YOLOv5 model with OpenCV, and it communicates with the Arduino over USB to control the motor through an L298N motor driver.

## Overview

The RC car leverages the following components:
- **YOLOv5 and OpenCV**: Used for detecting stop signs and lane lines in the video feed captured by a USB camera.
- **NVIDIA Jetson**: Processes the video feed to perform lane detection and stop sign detection, and directly controls a steering servo via the Jetson.GPIO library.
- **Arduino**: Controls the motor via an L298N motor driver. The Arduino receives commands from the Jetson when a stop sign is detected.

## Features

- **Lane Detection and Steering Control**: Processes the video feed to calculate steering angles based on lane line detection and controls a servo motor accordingly.
- **Stop Sign Detection**: Utilizes YOLOv5 to detect stop signs in real time.
- **Integrated Single Video Feed Processing**: Both lane detection and stop sign detection are performed on the same video stream.
- **Serial Communication**: The Jetson communicates with the Arduino via USB to send motor control commands.

## Requirements

### Hardware

- **NVIDIA Jetson** 
- **Arduino Uno**
- **L298N Motor Driver** 
- **Steering Servo Motor**
- **USB Camera**
- **RC Car Chassis with Wheels**

### Software

- Python 3
- YOLOv5 model
- Jetson.GPIO library
- Arduino IDE
- OpenCV 
- Pytorch
  
## Setup and Usage

### 1. Setting Up the NVIDIA Jetson

1. **Install Python 3 and Required Libraries**
   (You may need to use different steps to build OpenCV and PyTorch with CUDA for your machine):
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   pip3 install opencv-python torch torchvision jetson-gpio
   ```
3. **Clone the YOLOv5 Repository**:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   ```
4. **Copy the Python Scripts**:
   ```bash
   git clone https://github.com/sbudds/self_driving_rc-car.git
   ```
### 2. Setting Up the Arduino

1. **Hardware Connections**:
   - Connect the L298N motor driver to the Arduino and motor to motor driver
2. **Upload Arduino Code**:
   - Upload main.ino to arduino

### 3. Connecting the Components

- **Jetson to Arduino**: Connect via USB cable.
- **Camera**: Attach the  camera to  Jetson.
- **Steering Servo**: Connect the servo motor to the designated GPIO pin on the Jetson (this pin MUST have PWM support).
- **Power**: Ensure that the NVIDIA Jetson, Arduino, motor driver, and servo motor are all powered as per their specifications. My setup used two batteries; one 7.2V battery for the motors (through motor driver) and one 12V battery for the NVIDIA Jetson. The arduino recieved power via USB from Jetson. 

### 4. Running the System

1. **Launch the Integrated Script**:
   ```bash
   python3 main_detection.py
   ```
2. **System Operation**:
   - The system processes the video feed to perform lane detection and compute the steering angle.
   - Simultaneously, it detects stop signs using YOLOv5.
   - The calculated steering angle controls the servo motor via the Jetson’s GPIO.
   - Upon detecting a stop sign, the system sends a `STOP` command to the Arduino to halt the motor. 
   - Press `q` in the display window to exit the program.

### 5. Arduino Motor Control

- **Normal Operation**: The motor runs continuously.
- **Stop Sign Detected**: Reacts to signals sent from NVIDIA Jetson 
  
## License

This project is open-source and available under the [MIT License](LICENSE).

---


