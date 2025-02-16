#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
String gesture = "";
unsigned long lastUpdateTime = 0;
const unsigned long UPDATE_INTERVAL = 100; // Minimum time between updates (ms)

void setup() {
  lcd.init();
  lcd.backlight();
  Serial.begin(9600);
  
  lcd.setCursor(0, 0);
  lcd.print("Gesture:");
}

void loop() {
  if (Serial.available() > 0 && millis() - lastUpdateTime > UPDATE_INTERVAL) {
    String newGesture = Serial.readStringUntil('\n');
    newGesture.trim();
    
    // Label akan berubah jika gesture baru tidak sama dengan gesture sebelumnya
    if (newGesture != gesture) {
      gesture = newGesture;
      
      // Membersihkan layar baris kedua
      lcd.setCursor(0, 1);
      lcd.print("                ");
      
      // Center and display new gesture
      int pos = (16 - gesture.length()) / 2;
      lcd.setCursor(pos, 1);
      lcd.print(gesture);
      
      lastUpdateTime = millis();
    }
    
    // Clear serial buffer
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}