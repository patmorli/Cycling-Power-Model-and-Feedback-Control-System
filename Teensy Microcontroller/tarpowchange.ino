void tarpowchange(){

switch (tarpowcounter) {
     case 2:
    tarpow = basetarpow;
    //Serial.println("case2");
    break;
    case 3:
    tarpow = basetarpow*1.075;
    //Serial.println("case3");
    break;
    case 4:
    tarpow = basetarpow*0.925;
    //Serial.println("case4");
    break;
    case 5:
    tarpow = basetarpow*0.85;
    break;
    case 6:
    tarpow = basetarpow;
    break;
    case 7:
    tarpow = basetarpow*0.925;
    break;
    case 8:
    tarpow = basetarpow*1.075;
    break;
    case 9:
    tarpow = basetarpow*1.15;
    break;
    case 10:
    tarpow = basetarpow;
    break;
    case 11:
    tarpow = basetarpow*0.85;
    break;
    case 12:
    tarpow = basetarpow*0.925;
    break;
    case 13:
    tarpow = basetarpow*1.075;
    break;
    case 14:
    tarpow = basetarpow;
    break;
    case 15:
    tarpow = basetarpow*1.15;
    break;
    case 16:
    tarpow = basetarpow*1.075;
    break;
    case 17:
    tarpow = basetarpow*0.925;
    break;
    case 18:
    tarpow = basetarpow;
    break;
    case 19:
    tarpow = 5;
    break;
}
}
