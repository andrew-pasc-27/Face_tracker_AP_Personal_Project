#include <Servo.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>

const byte PAN_PIN = 10;
const byte TILT_PIN = 9;

Adafruit_SSD1306 display(128, 64, &Wire, -1);

Servo panServo;
Servo tiltServo;
String serialLine;

void setup() {
  Serial.begin(115200);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)){
    Serial.println(F("screen failed"));
    for (;;); //continue without the screen :(
  }
  
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(90);
  tiltServo.write(90);

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println(F(" Welcome =) "));
  display.display();
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

  pan = constrain(pan, 10, 170);
  tilt = constrain(tilt, 10, 170);
  panServo.write(pan);
  tiltServo.write(tilt);

  display.clearDisplay();
  display.setCursor(0,0);
  display.print("Pan: ");
  display.println(pan);
  display.print("Tilt: ");
  display.println(tilt);
  display.display();
}
