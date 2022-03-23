#define DEPLOYMENT_PIN 3
#define SWITCH_PIN 8

void setup() {
  // put your setup code here, to run once:
  pinMode(DEPLOYMENT_PIN, OUTPUT);       // set pin as main charge
  digitalWrite(DEPLOYMENT_PIN, LOW);
  pinMode(SWITCH_PIN, INPUT); // set as the switch input
  Serial.begin(115200);x
  while (!Serial);
}

void loop() {
  // put your main code here, to run repeatedly:

  if (digitalRead(SWITCH_PIN) == HIGH) {
    digitalWrite(DEPLOYMENT_PIN, HIGH);
  } else {
    digitalWrite(DEPLOYMENT_PIN, LOW);
  }

  delay(10);

}
