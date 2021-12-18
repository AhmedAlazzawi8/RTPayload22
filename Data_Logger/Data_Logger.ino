#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "Adafruit_BMP3XX.h"

#define SEALEVELPRESSURE_HPA (1013.25)

Adafruit_BMP3XX bmp;
Adafruit_BNO055 bno = Adafruit_BNO055(55);

unsigned long clocktime;

void setup(void) 
{
  Serial.begin(9600);
  Serial.println("Orientation Sensor Test");
  
  /* Initialise the sensor */
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }

  Serial.println("Adafruit BMP388 / BMP390 test");

  if (!bmp.begin_I2C()) {   // hardware I2C mode, can pass in address & alt Wire
    //if (! bmp.begin_SPI(BMP_CS)) {  // hardware SPI mode  
    //if (! bmp.begin_SPI(BMP_CS, BMP_SCK, BMP_MISO, BMP_MOSI)) {  // software SPI mode
    Serial.println("Could not find a valid BMP388 sensor, check wiring!");
    while (1);
  }

  // Set up oversampling and filter initialization
  bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_8X);
  bmp.setPressureOversampling(BMP3_OVERSAMPLING_4X);
  bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
  bmp.setOutputDataRate(BMP3_ODR_50_HZ);

  Serial.println("Time:\t\tX:\t\tY:\t\tZ:\t\tAlt.:\t\tTemp.:\t\tPre.:");
  
  delay(1000);
    
//  bno.setExtCrystalUse(true);
}

void loop(void) 
{
  Serial.print(clocktime);
  Serial.print("\t\t");
  // BMP DATA
  /* Get a new sensor event */ 
  sensors_event_t event; 
  bno.getEvent(&event);
  
  /* Display the floating point data */
  Serial.print(event.orientation.x, 4);
  Serial.print("\t\t");
  Serial.print(event.orientation.y, 4);
  Serial.print("\t\t");
  Serial.print(event.orientation.z, 4);
  Serial.print("\t\t");

  // BMP DATA
  if (! bmp.performReading()) {
    Serial.println("Failed to perform reading :(");
    return;
  }
  //Serial.print("Temperature = ");
  Serial.print(bmp.temperature);
  Serial.print("\t\t");

  //Serial.print("Pressure = ");
  Serial.print(bmp.pressure / 100.0);
  Serial.print("\t\t");

  //Serial.print("Approx. Altitude = ");
  Serial.print(bmp.readAltitude(SEALEVELPRESSURE_HPA));
  Serial.println("\t\t");

  Serial.println("");
  
  clocktime = millis();
  delay(250);
}
