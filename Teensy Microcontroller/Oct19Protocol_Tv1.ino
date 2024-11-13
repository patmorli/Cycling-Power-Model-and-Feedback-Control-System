/*Protocol1_Tv2, Mayerhofer Patrick, May, 2019
 * Last Modifications: April, 2021
 * reads torque data from crank via fm sensor, angular velocity,
 * calculates mean power, outputs metronome beep, reads gps data,
 * saves data at SD, 
 * 
 * If SD does not work, try SD example, dont forget to change
 * channel to 10
 * 
 * For basebpm calculations: can start whenever
 * For open loop experiment: back spin crank to turn on torque 
 * device, put horizontal, wait until metronome gets fast, start
*/

 /* Box Input:
1: GPS (oben) - teensy box on top, green/blue
2: Switch GND -white 
3: Shield GND - black thick
4: GPS Ground - yellow/black
5: Hz1 - red
6: Hz2 - black thin
7: Switch - green thin
*/


// Simple read HZ using pin 11 and interrupts
//Connect red (Blue) to 3.3v
//Black (green) = ground (GND)
//White (violett) = hz signal into any digital pin, just change number below to match pin


/*libraries*/
#include <math.h>    // (no semicolon)
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <synth_simple_drum.h>
#include <SerialFlash.h>
#include <ADC.h>

/*pins*/
#define hzPin 3          // SRM signal
#define reedPin 16     // reed switch
//#define tonePin    44      // metronome
//const int storagePin = 10;  // automatically if using the adafruit shield
const int gpsPin = 17;  // input of gps raw data, between 0 and 5 Volt - 
//always use voltage divider with arduino due!!!

#define SDCARD_CS_PIN    10
#define SDCARD_MOSI_PIN  7
#define SDCARD_SCK_PIN   14


///changeable constants
const double maxSpeed = 35.0; //changeable with Vbox software
const float pi = 3.14;
const double basebpm = 74.0;
const short toneFreq = 800;   // tone starting value, Hz
char storeName[11] = "chf1.txt";
uint32_t trialTime = 60000; // length of step change trials
File SDdata;





/*for metronome*/   

volatile uint8_t startingFlag = 0;   //change to 1 to do metronome tests                     
double previousMillis = 0; 
double currentMillis = 0; 
double beepDif; // time between previous and next beap
double basetimeDif = (500/(basebpm/60)); //from bpm to ms
unsigned long lasttrialTime = 0; //for step change trials
unsigned short bpmcounter = 1;
float bpm = 10.0000;
double a1 = basetimeDif * 0.25;  //amplitude
double a2 = basetimeDif * 0.1;  //amplitude
int startTime;
int offsetTime = 0; //to start chirp signal time in end of experiment from zero

/*For the metronome sound*/
AudioSynthSimpleDrum     drum3;          //xy=424,310
AudioMixer4              mixer1;         //xy=737,265
AudioOutputI2S           i2s1;           //xy=979,214
AudioConnection          patchCord2(drum3, 0, mixer1, 2);
AudioConnection          patchCord5(mixer1, 0, i2s1, 0);
AudioConnection          patchCord6(mixer1, 0, i2s1, 1);
AudioControlSGTL5000     sgtl5000_1;     //xy=930,518


/* for angvel and power */
 volatile uint8_t angvelFlag = 0;
 volatile uint8_t angvelcopyFlag = 0;
 unsigned long t1Shared;
 unsigned int tDifShared;
 
 
 

/* for counterhz */
// shared variables are updated by the ISR and read by loop.
// const int BUFFERSIZE = 50;//change this to whatever size you want to read from the serial port
// int HZInShared[BUFFERSIZE];
volatile uint8_t counterFlag = 0;
double HZtConstantSumShared;
unsigned int HZCounterShared;
unsigned long HZStart;
unsigned long HZPeriod;
unsigned long HZ;

/*for zerohz, if you use the button, put buttonStates to 0 and fill in the interrupt function*/
volatile uint8_t buttonState1 = 1;  
volatile uint8_t buttonState2 = 1;  
unsigned short zeroHZ = 0;
unsigned int zeroCounter = 0; // this variable appears in counterhz_interrupt
volatile uint8_t zeroFlag = 0; 

//Set up ADC
ADC *adc = new ADC(); // adc object;


void setup(void)
{
  //Serial.begin(4800);   // change if need more 
  pinMode(hzPin, INPUT);  
  pinMode(reedPin, INPUT_PULLUP);
  pinMode(gpsPin, INPUT);
  attachInterrupt(hzPin, counterhz, RISING);    // HZ of crank
  attachInterrupt(reedPin, reedtrigger, RISING);  // reed switch
  analogReadResolution(12); //to always have 4096 bits, was weird without, maybe possible to even put higher resolution

    adc->adc0->setAveraging(12); // set number of averages
    adc->adc0->setResolution(12); // set bits of resolution

    // it can be any of the ADC_CONVERSION_SPEED enum: VERY_LOW_SPEED, LOW_SPEED, MED_SPEED, HIGH_SPEED_16BITS, HIGH_SPEED or VERY_HIGH_SPEED
    // see the documentation for more information
    // additionally the conversion speed can also be ADACK_2_4, ADACK_4_0, ADACK_5_2 and ADACK_6_2,
    // where the numbers are the frequency of the ADC clock in MHz and are independent on the bus speed.
    adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::LOW_SPEED); // change the conversion speed
    // it can be any of the ADC_MED_SPEED enum: VERY_LOW_SPEED, LOW_SPEED, MED_SPEED, HIGH_SPEED or VERY_HIGH_SPEED
    adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::MED_SPEED); // change the sampling speed



  
 //bpm-millis calculation
    beepDif = basetimeDif;
    Serial.flush();

/* initialize SD card */

  //while (!Serial) { //comment in for computer connection
    ; // wait for serial port to connect. Needed for native USB port only
  //}


  //Serial.print("Initializing SD card...");

  // see if the card is present and can be initialized:
   
   SPI.setMOSI(SDCARD_MOSI_PIN);
  SPI.setSCK(SDCARD_SCK_PIN);
  if (!(SD.begin(SDCARD_CS_PIN))) {
    // stop here, but print a message repetitively
    while (1) {
      //Serial.println("Unable to access the SD card");
      delay(500);
    }
  }
  //Serial.println("card initialized.");
  
  /*teensy metronome adjustments*/
  AudioMemory(15);
  drum3.frequency(500);
  drum3.length(120);
  //drum3.secondMix(1.0);
  drum3.pitchMod(0.5);

  
  sgtl5000_1.enable();
  sgtl5000_1.volume(0.7);

  beepDif = 30000.00/bpm;
  
  }

void loop(void)
{
//output
String dataString = "";

/*for power calculations */
  static double HZtConstantSum;
  static unsigned int HZCounter;
  static unsigned int HZAve;
  static double torqueAve;
  static unsigned long t1;

  /*for metronome bpm*/

  /*for gps*/
  double gpsBitvolt;
  double gpsSpeed;

  
  // code for frequent metronome with millis() and also for bpm change
  currentMillis = millis();
  if (currentMillis - previousMillis >= beepDif){
     drum3.noteOn(); // play drum note whenever time has passed
     previousMillis = currentMillis; //for next round 
 
  // step changes
   if (currentMillis - lasttrialTime >= trialTime&&bpmcounter < 21&&startingFlag){
      
      // change bpm for step change
      //bpmcounter++;
      bpmcounter = 2;
      bpmchange();
      beepDif = round(30000/float(bpm));
      lasttrialTime = currentMillis;
      }
    /*
      if (bpmcounter >= 20){
        if(bpmcounter == 20){
          offsetTime = millis();
          bpmcounter++;
          }
      // this is for chirp:    
      //beepDif = a2*sin(2*pi*exp((currentMillis-offsetTime)/(400000/3))) + basetimeDif;      
      
  } 
*/
}
  
    
  
 

  noInterrupts(); // turn interrupts off quickly while we take local copies of the shared variables (because during calculations, interrupts will not stop running)
    if (angvelcopyFlag || zeroFlag){
      HZtConstantSum = HZtConstantSumShared;
      HZCounter = HZCounterShared;
      t1 = t1Shared;
      //Serial.println("no interrupts point");
      //Serial.println(tDifShared);
      
      /*Serial.println("HZSumShared");
      Serial.println(HZSumShared);
      Serial.println("HZCounterShared");
      Serial.println(HZCounterShared);  */
      
      //Set shared variables to 0, so that interrupt can start again
    
     HZtConstantSumShared = 0;
     HZCounterShared = 0;
     interrupts();

     

    //to not trigger the calculations before the copying was done
     if(angvelcopyFlag){
    angvelFlag = 1; }
      }
      

      // zero hz
    if (zeroFlag){
      zeroHZ = round((float(HZCounter) / HZtConstantSum));
      zeroFlag = 0;
      startingFlag = 1;
      startTime = millis();
      }
    
 
/*// for debugging
  if (counterFlag){
     counterFlag = 0;  // set 0 to be false again
     HZFlagCounterShared = 0;  // after 100 it will take copies again
     Serial.println(HZCounterShared); //debug
     Serial.println(HZSumShared);
   
    }*/

  // when reed triggers->calculations 
  if (angvelFlag){
      HZAve = round((float(HZCounter)/HZtConstantSum)-zeroHZ);
      torqueAve = double(HZAve)/22.0;

      //gps
       gpsBitvolt = analogRead(gpsPin); 
       gpsSpeed = (maxSpeed / 2.99) * ((3.3/4096.0)*gpsBitvolt); 


      // save all the fun
       dataString = (String(gpsSpeed) + ", "  + String(t1) + ", " + String(torqueAve)) + ", " + String(beepDif);
       //Serial.println(dataString);
        
      File SDdata = SD.open(storeName, FILE_WRITE);
      
      if (SDdata) {
        SDdata.println(dataString);
        SDdata.close();
        // print to the serial port too:
        }
      // if the file isn't open, pop up an error:
      else {
        SDdata.println("error opening datalog.txt");
        }
        
      //loeschen wenn alles passt 
     // Serial.println(beepDif); 
     // Serial.println(t1);
     // Serial.println(torqueAve); //with time constants
      
      
      // set back to not repeat this action
      angvelFlag = 0;
      angvelcopyFlag = 0;
      
    }

    
delay(1);
  

    }





  
