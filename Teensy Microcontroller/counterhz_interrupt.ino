void counterhz() //This is what determines the frequency

{
  HZPeriod = micros() - HZStart;
  if (HZPeriod >= 75) { //285us=3500hz, 334=3000hz
    HZStart = micros();

    HZ = 1000000 / HZPeriod; //convert period to frequency

    if (HZ > 300 && HZ < 10000) {   //simple filter to eliminate small or large values  //sum((pi*ti)/ti)
      HZtConstantSumShared += 1/(double(HZ));
      HZCounterShared += 1;    //sum((pi*ti)/ti) -> pi*ti=1 -->(398*(1/398))

      if (buttonState1){
        HZtConstantSumShared = 0;
        HZCounterShared = 0;
        buttonState1 = 0;
        zeroHZ = 0;
      }

      if (HZCounterShared > 9999 && buttonState2){
        zeroFlag = 1;
        buttonState2 = 0;
        }
      
        

      
       
    
        }
      
      }
  }
