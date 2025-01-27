# Self-Driving RC Car with YOLOv5 and OpenCV

This project is a self-driving RC car that uses a Raspberry Pi for image processing and an Arduino for motor control. The system is capable of detecting stop signs using a YOLOv5 model with OpenCV, and it communicates with the Arduino to control the motors through an L298N motor driver.

## Overview
The RC car leverages the following components:
- **YOLOv5 and OpenCV**: Used for detecting stop signs in the video feed captured by a USB camera mounted on the RC car.
- **Raspberry Pi**: Handles video processing and communicates with the Arduino via serial.
- **Arduino**: Controls the motor through an L298N motor driver, stopping the motor when a stop sign is detected.

## Features
- **Object Detection**: Uses YOLOv5 to detect stop signs in real time.
- **Video Input**: Captures video from a USB camera.
- **Motor Control**: Automatically stops the motor when a stop sign is detected.
- **Serial Communication**: Raspberry Pi sends commands to the Arduino via USB.

## Requirements
### Hardware
- Raspberry Pi (tested on Raspberry Pi 4)
- Arduino Uno
- L298N Motor Driver
- Single DC motor
- USB camera
- RC car chassis with wheels

### Software
- Python 3
- OpenCV
- PyTorch
- YOLOv5 model
- Arduino IDE

## Setup and Usage

### 1. Setting Up the Raspberry Pi
1. Install Python 3 and the required libraries:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   pip3 install opencv-python torch torchvision
   ```
2. Clone the YOLOv5 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   ```
3. Copy the Python script for stop sign detection to the Raspberry Pi.

### 2. Setting Up the Arduino
1. Connect the L298N motor driver to the Arduino and motor.
2. Upload the Arduino code to the Arduino Uno using the Arduino IDE.

### 3. Connecting the Components
- Connect the Raspberry Pi to the Arduino via a USB cable.
- Ensure the camera is connected to the Raspberry Pi.
- Power the motor and L298N motor driver as per the circuit.

### 4. Running the System
1. Start the Python script on the Raspberry Pi to begin video processing and detection.
   ```bash
   python3 stop_sign_detection.py
   ```
2. The Raspberry Pi will detect stop signs and send a `STOP` command to the Arduino via serial communication.

### 5. Arduino Motor Control
- By default, the motor runs continuously.
- When a `STOP` signal is received, the motor will halt.

## Circuit Diagram
- Connect the L298N motor driver to the Arduino pins as follows:
  - EN: PWM control pin
  - IN1: Motor direction control
  - IN2: Motor direction control
- Connect the motor to the output terminals of the L298N motor driver.
- Power the Arduino, motor driver, and Raspberry Pi appropriately.

## Contributions
Contributions are welcome! Feel free to submit a pull request or open an issue.

## License
This project is open-source and available under the [MIT License](LICENSE).

