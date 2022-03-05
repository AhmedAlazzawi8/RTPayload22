#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include "Adafruit_BMP3XX.h"
#include <utility/imumaths.h>

#define APOGEE 1                       // Test apogee
#define SEALEVELPRESSURE_HPA (109.25) // This needs to be set for the location and day
#define SOME_AMOUNT_I_DONT_REALLY_CARE  2 // meters

#define DEPLOYMENT_PIN 2

//#define DEBUG_PRINT

Adafruit_BNO055 bno = Adafruit_BNO055(55);
Adafruit_BMP3XX bmp1; 
static float x_rotation, y_rotation, z_rotation;
// Initialize altitude
static float offset, altitude;

void setup(void) 
{
  Serial.begin(115200);
  Serial.println("Orientation Sensor Test"); Serial.println("");

  pinMode(DEPLOYMENT_PIN, OUTPUT);
  digitalWrite(DEPLOYMENT_PIN, LOW);

  
  /* Initialise the sensor */
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
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
  
  
  // set offset
  offset = bmp1.readAltitude(SEALEVELPRESSURE_HPA);
  
  delay(1000);
    
  bno.setExtCrystalUse(true);
}

int v_magnitude = 0;

float ref_X;
float ref_Y;
float ref_Z;
int counter = 0;

void loop(void) 
{
  
  /* Get a new sensor event */ 
  sensors_event_t event; 
  bno.getEvent(&event);
  
  if(counter == 0)
  {
    ref_X = (event.orientation.x > 180)? event.orientation.x - 360: event.orientation.x;
    ref_Y = (event.orientation.y > 180)? event.orientation.y - 360: event.orientation.y;
    ref_Z = (event.orientation.z > 180)? event.orientation.z - 360: event.orientation.z;
    counter++;
    
#ifdef DEBUG_PRINT
    Serial.print("Ref_x:");
    Serial.print(ref_X, 4);
    Serial.print("Ref_y:");
    Serial.print(ref_Y, 4);
    Serial.print("Ref_z:");
    Serial.print(ref_Z, 4);
    Serial.print("\n");
    //Serial.print("X: %f, Y %f, Z %f",ref_X, ref_Y, ref_Z);
#endif
  }
  
  x_rotation = (event.orientation.x > 180)? event.orientation.x - 360: event.orientation.x;
  y_rotation = (event.orientation.y > 180)? event.orientation.y - 360: event.orientation.y;
  z_rotation = (event.orientation.z > 180)? event.orientation.z - 360: event.orientation.z;
  
  altitude = (bmp1.readAltitude(SEALEVELPRESSURE_HPA) - offset);

  if( (((abs(y_rotation - ref_Y) <= 5) && (abs(y_rotation - ref_Y) >= -15)) || ((abs(x_rotation - ref_X) <= 5) && (abs(x_rotation - ref_X) >= -15)))
  
  && (altitude > SOME_AMOUNT_I_DONT_REALLY_CARE))
  {
    digitalWrite(DEPLOYMENT_PIN, HIGH);
    delay(10000);
    digitalWrite(DEPLOYMENT_PIN,LOW);
    while(1);
  }
  
#ifdef DEBUG_PRINT
  Serial.print("\tY: ");
  Serial.print(y_rotation, 4);
  Serial.print("\tZ: ");
  Serial.print(z_rotation, 4);
  Serial.println("");
#endif
   
  delay(10);
}
