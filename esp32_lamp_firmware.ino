#include <Arduino.h>
#include <Wire.h>

const int LED_PIN = 25;          // External LED: GPIO 25 (D25) -> resistor -> LED -> GND.
const int BUILTIN_LED_PIN = 2;   // Common ESP32 DevKit onboard LED.
const int PIR_PIN = 27;
const int I2C_SDA_PIN = 21;
const int I2C_SCL_PIN = 22;
const uint8_t BH1750_ADDRESS = 0x23;
const int PWM_FREQ = 5000;
const int PWM_RESOLUTION = 8;   // 8-bit duty cycle: 0..255
const int DEFAULT_BRIGHTNESS_PERCENT = 70;
const unsigned long SENSOR_INTERVAL_MS = 1000;

unsigned long lastSensorRead = 0;

void setBrightnessPercent(int percent) {
  percent = constrain(percent, 0, 100);
  int duty = map(percent, 0, 100, 0, 255);
  ledcWrite(LED_PIN, duty);
  ledcWrite(BUILTIN_LED_PIN, duty);
  Serial.print("LED brightness set to ");
  Serial.print(percent);
  Serial.println("%");
}

bool readLux(float &lux) {
  Wire.beginTransmission(BH1750_ADDRESS);
  Wire.write(0x10);
  if (Wire.endTransmission() != 0) {
    return false;
  }

  delay(180);
  Wire.requestFrom(BH1750_ADDRESS, static_cast<uint8_t>(2));
  if (Wire.available() < 2) {
    return false;
  }

  uint16_t raw = (static_cast<uint16_t>(Wire.read()) << 8) | Wire.read();
  lux = raw / 1.2f;
  return true;
}

void readBrightnessCommand() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();
  const String prefix = "BRIGHTNESS:";
  if (line.startsWith(prefix)) {
    setBrightnessPercent(line.substring(prefix.length()).toInt());
    Serial.println("OK");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT_PULLDOWN);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUILTIN_LED_PIN, OUTPUT);
  ledcAttach(LED_PIN, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(BUILTIN_LED_PIN, PWM_FREQ, PWM_RESOLUTION);
  setBrightnessPercent(DEFAULT_BRIGHTNESS_PERCENT);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Serial.println("ESP32 sensor controller ready");
}

void loop() {
  readBrightnessCommand();

  if (millis() - lastSensorRead < SENSOR_INTERVAL_MS) {
    return;
  }
  lastSensorRead = millis();

  float lux = 0;
  if (!readLux(lux)) {
    Serial.println("Sensor read error");
    return;
  }

  int motion = digitalRead(PIR_PIN) == HIGH ? 1 : 0;
  Serial.print("LUX:");
  Serial.print(lux, 2);
  Serial.print(",MOTION:");
  Serial.println(motion);
}
