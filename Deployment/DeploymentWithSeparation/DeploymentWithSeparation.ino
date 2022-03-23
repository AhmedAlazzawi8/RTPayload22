#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include "Adafruit_BMP3XX.h"
#include <utility/imumaths.h>


// uncomment the line below to print to the serial monitor
#define DEBUG 

// defining pins (pin numbers need to be set correctly)
#define SEPARATION_PIN          7
#define DEPLOYMENT_MAIN_PIN     8
#define DEPLOYMENT_BACKUP_PIN   9
#define LED_PIN                 13

// apogee and BMP related defines
#define APOGEE_PREDICTED 1460         // Predicted apogee in meters
#define CROSSOVER_HEIGHT 1400         // Threshold to crossover before the payload can be deployed              
#define SEALEVELPRESSURE_HPA (109.25) // This needs to be set for the location and day

// timing 
#define dT                      10 // this is 10ms, the period of the main loop
#define CHUTE_DELAY_TIME      1500 // how long to wait after sensing separation
#define EMATCH_DELAY_TIME     2500 // how long to keep the deployment pins high


// static variables
static Adafruit_BMP3XX bmp;                // BMP388 object
static float offset;                       // offset from the initial reading we have from the bmp
static float altitude;                     // our known altitude
static unsigned long prev_time = 0;        // keeps track of last time the main loop was run 
static unsigned long curr_time = millis(); // keeps track of the current millisecond time  


void setup() {
  
  // setup pins
  pinMode(SEPARATION_PIN, INPUT);
  pinMode(DEPLOYMENT_MAIN_PIN, OUTPUT);
  pinMode(DEPLOYMENT_BACKUP_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(DEPLOYMENT_MAIN_PIN, LOW);
  digitalWrite(DEPLOYMENT_BACKUP_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  delay(100);

#ifdef DEBUG
  Serial.begin(115200);
  Serial.println("This is the deployment software v2.0 using 2 BMPs and separation sensing"); 
  Serial.println("Pins have been configured for seperation sensing and dual deployment");
#endif

  // setup BMP sensor
  if (!bmp.begin_I2C()) {   // hardware I2C mode, can pass in address & alt Wire
                            // uint8_t addr=BMP10XX_DEFAULT_ADDRESS, TwoWire *theWire=&Wire
    #ifdef DEBUG
    Serial.println("Could not find a valid BMP388 sensor, check wiring!");
    #else
    digitalWrite(LED_PIN, HIGH);
    while (1);
  }

  // Set up oversampling and filter initialization
  bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_8X);
  bmp.setPressureOversampling(BMP3_OVERSAMPLING_4X);
  bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
  bmp.setOutputDataRate(BMP3_ODR_50_HZ);

  // let sensor setup
  delay(2000);
  // read out garbage
  altitude = bmp.readAltitude(SEALEVELPRESSURE_HPA);
  delay(500);  
  // set offset
  offset = bmp.readAltitude(SEALEVELPRESSURE_HPA);

  // Blink LED to indicate things are working
  digitalWrite(LED_PIN, HIGH);
  delay(1000);
  digitalWrite(LED_PIN, LOW);
}

void loop() {

  curr_time = millis(); // update time
  if (curr_time - prev_time > dT){
    // update previous time
    prev_time = curr_time;

    // update altitude reading
    altitude = (bmp.readAltitude(SEALEVELPRESSURE_HPA) - offset);

    // check deployment conditions
    if ((digitalRead(SEPARATION_PIN) == LOW) && (altitude >= CROSSOVER_HEIGHT)) {
      // delay for the chute deployment
      delay(CHUTE_DELAY_TIME);
      
      // ignite main charge
      digitalWrite(DEPLOYMENT_MAIN_PIN, HIGH);
      delay(EMATCH_DELAY_TIME);
      digitalWrite(DEPLOYMENT_MAIN_PIN, LOW);
      
      // ignite backup charge
      digitalWrite(DEPLOYMENT_BACKUP_PIN, HIGH);
      delay(EMATCH_DELAY_TIME);
      digitalWrite(DEPLOYMENT_BACKUP_PIN, LOW);

      // exit
      while(1);
    }    
  }
}
