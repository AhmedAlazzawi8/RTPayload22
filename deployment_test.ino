#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BMP3XX.h"

#define APOGEE 1                       // Test apogee
#define SEALEVELPRESSURE_HPA (109.25) // This needs to be set for the location and day

// Make two bmp objects of class Adafruit_BMP3XX
Adafruit_BMP3XX bmp1; 

// Initialize altitude
static float offset, altitude;

void setup() {
  pinMode(10, OUTPUT);       // set pin 9 as main charge;
  digitalWrite(10, HIGH);
  Serial.begin(115200);
  while (!Serial);
  
  Serial.println("Adafruit BMP1088");

  if (!bmp1.begin_I2C()) {   // hardware I2C mode, can pass in address & alt Wire
                            // uint8_t addr=BMP10XX_DEFAULT_ADDRESS, TwoWire *theWire=&Wire
    Serial.println("Could not find a valid BMP10 sensor, check wiring!");
    while (1);
  }

  // Set up oversampling and filter initialization
  bmp1.setTemperatureOversampling(BMP3_OVERSAMPLING_8X);
  bmp1.setPressureOversampling(BMP3_OVERSAMPLING_4X);
  bmp1.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
  bmp1.setOutputDataRate(BMP3_ODR_50_HZ);

  // let sensor setup
  delay(2000);
  
  // read out garbage
  altitude = bmp1.readAltitude(SEALEVELPRESSURE_HPA);
  delay(500);
  
  // read initial
  altitude = bmp1.readAltitude(SEALEVELPRESSURE_HPA);
  delay(500);
  
  // set offset
  offset = bmp1.readAltitude(SEALEVELPRESSURE_HPA);
  
}

void loop() {
  if (! bmp1.performReading()) {
    Serial.println("Failed to perform reading for sensor1");
    return;
  }
  
  // read sensors
  altitude = (bmp1.readAltitude(SEALEVELPRESSURE_HPA) - offset);
 Serial.println(altitude);

  // compare with apogee
  if(altitude >= APOGEE){
      digitalWrite(10, HIGH);
      while(1);
  }
  
  delay(100);
}
