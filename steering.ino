#include <Servo.h>

Servo steeringServo;

void setup() {
  steeringServo.attach(11); // Connect the servo signal pin to pin 11
  Serial.begin(9600);
  steeringServo.write(105);  // Set the initial angle
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); // Read the incoming angle until newline
    Serial.println(input);  // Print the received input for debugging
    int angle = input.toInt(); // Convert the input to an integer
    if (angle >= 80 && angle <= 130) { // Check if the angle is within the valid range
      steeringServo.write(angle); // Set the servo to the angle
    }
  }
}
