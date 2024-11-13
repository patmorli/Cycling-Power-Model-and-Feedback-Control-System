void reedtrigger(){
  static unsigned long t2 = 0;
  t1Shared = millis(); 
  tDifShared = t1Shared - t2;
  if (tDifShared>150){
  angvelcopyFlag = 1;
  angvelFlag = 1;
  t2 = t1Shared; }
 
  }
