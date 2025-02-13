import Jetson.GPIO as GPIO
import time

# Set up GPIO for servo control
SERVO_PIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(0)

def angle_to_duty_cycle(angle):
    """Convert angle (in degrees) to duty cycle percentage for the servo."""
    return (angle / 18.0) + 2.5

def set_steering_angle(angle):
    """Set the servo to the specified angle."""
    duty_cycle = angle_to_duty_cycle(angle)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.2)  # small delay to let the servo reach the angle
    servo.ChangeDutyCycle(0)  # stop the PWM signal

try:
    while True:
        user_input = input("Enter angle (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        try:
            angle = float(user_input)
        except ValueError:
            print("Please enter a valid number for the angle.")
            continue

        print(f"Setting servo to {angle} degrees.")
        set_steering_angle(angle)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    servo.stop()
    GPIO.cleanup()
