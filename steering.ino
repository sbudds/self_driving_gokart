#include <Servo.h>

Servo steeringServo;

void setup() {
  steeringServo.attach(11); // Connect the servo signal pin to pin 9
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\\n'); // Read the incoming angle
    int angle = input.toInt(); // Convert the input to an integer
    if (angle >= 80 && angle <= 130) { // Check if the angle is within the valid range
      steeringServo.write(angle); // Set the servo to the angle
    }
  }
}
