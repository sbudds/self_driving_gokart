#include <Servo.h>

// Motor control pins
const int EN = 10;  // Enable pin for motor
const int IN1 = 9;  // Input 1 for motor
const int IN2 = 8;  // Input 2 for motor

// Steering servo
Servo steeringServo;

// Initialize setup
void setup() {
  // Set motor control pins as outputs
  pinMode(EN, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Start serial communication
  Serial.begin(9600);

  // Initialize motor in the forward direction by default
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(EN, 255);  // Motor speed ranges from 0-255

  // Attach the steering servo to pin 11
  steeringServo.attach(11);
  steeringServo.write(105);  // Set initial servo angle
}

// Main loop
void loop() {
  // Check if data is available on the serial port
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read the incoming command

    if (command == "STOP") {
      // Stop the motor
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      analogWrite(EN, 0);  // Disable motor
      Serial.println("Motor stopped");
    }

    else if (command == "GO") {
      // Start the motor
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      analogWrite(EN, 255);  // Full speed
      Serial.println("Motor running");
    }

    // Handle steering commands (angle input)
    else {
      int angle = command.toInt();  // Convert input to integer
      if (angle >= 80 && angle <= 130) {  // Check if the angle is within a valid range
        steeringServo.write(angle);  // Set the steering servo to the angle
        Serial.print("Steering angle set to: ");
        Serial.println(angle);
      } else {
        Serial.println("Invalid angle. Please enter a value between 80 and 130.");
      }
    }
  }
}
