void bpmchange(){

   switch (bpmcounter) {
    case 2:
    bpm = basebpm;
    //Serial.println("case2");
    break;
    case 3:
    bpm = basebpm*1.05;
    //Serial.println("case3");
    break;
    case 4:
    bpm = basebpm*0.95;
    //Serial.println("case4");
    break;
    case 5:
    bpm = basebpm;
    break;
    case 6:
    bpm = basebpm*0.9;
    break;
    case 7:
    bpm = basebpm*1.1;
    break;
    case 8:
    bpm = basebpm*0.95;
    break;
    case 9:
    bpm = basebpm*1.05;
    break;
    case 10:
    bpm = basebpm*0.9;
    break;
    case 11:
    bpm = basebpm*1.1;
    break;
    case 12:
    bpm = basebpm;
    break;
    case 13:
     bpm = basebpm*1.05;
    break;
    case 14:
    bpm = basebpm*0.95;
    break;
    case 15:
    bpm = basebpm*1.05;
    break;
    case 16:
    bpm = basebpm*0.9;
    break;
    case 17:
    bpm = basebpm*1.1;
    break;
    case 18:
    bpm = basebpm*0.95;
    break;
    case 19:
    bpm = basebpm*1.05;
    break;
    case 20:
    bpm = 10.0;
    break;
}
}
