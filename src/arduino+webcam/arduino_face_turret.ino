#include <Servo.h>

// Change these pins to match the pins used by your turret.
const byte PAN_PIN = 10;
const byte TILT_PIN = 9;

Servo panServo;
Servo tiltServo;
String serialLine;

void setup() {
  Serial.begin(115200);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(90);
  tiltServo.write(90);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(serialLine);
      serialLine = "";
    } else if (c != '\r') {
      serialLine += c;
    }
  }
}

void handleCommand(const String &command) {
  int comma = command.indexOf(',');
  if (comma < 1) {
    return;
  }

  int pan = command.substring(0, comma).toInt();
  int tilt = command.substring(comma + 1).toInt();
  panServo.write(constrain(pan, 10, 170));
  tiltServo.write(constrain(tilt, 10, 170));
}
