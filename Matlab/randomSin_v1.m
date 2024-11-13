%randomSin_v1
%Started Jun 05, 2018
%Create Chirps and Sinosoids to have "random" bpm input 



%% chirp function
t = 0:1/1e3:2; 
y = chirp(t,0,1,250);
figure(1);
spectrogram(y,256,250,256,1e3,'yaxis');
figure(2);
plot(y);


%% random frequencies and amplitudes cosine wave
baselineCadence = 75;
amplitude = baselineCadence * 0.2;
trialtime = 2400; 

t = linspace(0, trialtime, 1000);  
% Make random frequencies.
frequency = .04 + .04 * rand(1, length(t));
% Construct waveform.
y = baselineCadence + amplitude * cos(2 .* pi .* frequency .* t);
plot(t, y, 'b-');
grid on;
% % Enlarge figure to full screen.
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);

%% Chirp signal by hand
Baseline = 80;
t = 0:0.01:100;    % time
A = Baseline*0.2;              % Amplitude
omega = 2;          % constant
s_t = -t.^2/16;      %fq modulation
s_t2 = t;
y_t = A*cos(omega*t+s_t) + Baseline;
plot(t, y_t);



%%

Baseline = 80;
c = 0.01;
t = 0:0.1:600 ;    % time
a = Baseline*0.2; % Amplitude

x1=[0,0];
x2=[300,10000];
x3=[600,0];
Y=[x1(2);x2(2);x3(2)];
A=[x1(1)^2 x1(1) 1;x2(1)^2 x2(1) 1;x3(1)^2 x3(1) 1];
X=inv(A)*Y;
x=x1(1):0.1:x3(1);
Y=X(1)*x.^2+X(2)*x;

y = a .* sin(Y*c) + Baseline;


plot(t, y);


%%
Baseline = 80;
c = 30;
t = 0:0.1:600 ;    % time
a = Baseline*0.2; % Amplitude
xe = -300:0.1:300;
Y = exp(0.00004.*-xe.^2); 
figure(1);
plot(t,Y);

y = a .* sin(Y*c) + Baseline;
plot(t, y);

%% loop for arduino
clear all;
Baseline = 80;
c = 30;
a = Baseline*0.2; % Amplitude
xe = -300;
for i = 1:100
  e(i) = exp(0.00004*-xe^2);
    xe = xe + 1;
end

plot(e);

for i = 1:100
    y(i) = a*sin(c*exp(-xe^2)) + Baseline;
    xe = xe + 1;
end

plot(y)


%% with log

xe = 1;

for i = 1:1000
    l(i) = -log(xe);
    xe = xe + 1;
end
plot(l);


clear all;
Baseline = 80;
basetimedif = 1/(80/60);
c = 30;
a = basetimedif*0.2; % Amplitude
xe = 1800;
time = 1800;
for i = 1:time
    l(i) = 10*log(xe);
    y(i) = a*sin(10*log(xe)) + basetimedif;
    xe = xe - 1;
end

plot(y)


%%
xe = 1;

for i = 1:180000
    l(i) = log(xe);
    xe = xe + 1;
end
plot(l);

%% final for arduino
clear all;
tic;
Baseline = 80;
basetimedif = 1/(80/60);
c = 30;
a = basetimedif*0.2; % Amplitude
xe = 1;
endtime = 60000;

for i = 1:endtime
    y(i) = a*sin(9*exp(2*i/endtime)) + basetimedif;
    xe = xe + 1;
end

plot(y)

%%
clear all;
tic;
Baseline = 86;
basetimedif = 500/(Baseline/60);
a = basetimedif*0.05; % Amplitude
xe = 1;
endtime = 1000000;

%while toc < endtime
for i = 1:endtime
    y2(xe) = a*sin(2*pi*exp(xe/(endtime/4))) + basetimedif;
    xe = xe + 1; 
end
figure
plot(y2)


%%
clear all;
tic;
Baseline = 80;
basetimeDif = 1000/(Baseline/60);
c = 30;
a = basetimeDif*0.2; % Amplitude
xe = 1;
endTime = 60000;

%while toc < endtime
i = 1;
delFreq = basetimeDif;
previousMillis = 0;
while toc*1000 < endTime
    currentMillis = toc * 1000;
if (currentMillis - previousMillis >= delFreq)
    %first number in brackets somehow equals the number of peaks
    %but just if the 2 inside the exp stays 2
    delFreq(i) = a*sin(5*exp(2*currentMillis/endTime)) + basetimeDif; 
    previousMillis = currentMillis;
    i = i+1;
end 
end
figure
plot(delFreq)

%% decreasing chirp
clear all;
tic;
Baseline = 80;
basetimeDif = 1000/(Baseline/60);
c = 30;
a = basetimeDif*0.2; % Amplitude
xe = 1;
endTime = 60000;

%while toc < endtime
i = 1;
delFreq = basetimeDif;
previousMillis = 0;
while toc*1000 < endTime
    currentMillis = toc * 1000;
%if (currentMillis - previousMillis >= delFreq)
    %first number in brackets somehow equals the number of peaks
    %but just if the 2 inside the exp stays 2
    delFreq(i) = a*currentMillis*sin(0.5*exp(2*currentMillis/endTime)) + basetimeDif; 
    previousMillis = currentMillis;
    i = i+1;
%end 
end
figure
plot(delFreq)

%% try
clear all;
tic;
endTime = 10000;
previousMillis = 0;
i = 1;
while toc*1000 < endTime
    currentMillis = toc * 1000;
if (currentMillis - previousMillis >= delFreq)
    %first number in brackets somehow equals the number of peaks
    %but just if the 2 inside the exp stays 2
    delFreq(i) = currentMillis;
    previousMillis = currentMillis;
    i = i+1;
end 
end
figure
plot(delFreq);

%% let's do the sum-of-sines wave
clear all
Baseline = 80;
basetimedif = 500/(Baseline/60);
a = basetimedif*0.1; % Amplitude
endtime = 1000000;
x = 1;
for i=1:endtime
    sin2(x) =  sin(0.002*pi*i*(1000/(5000*7))); %0.0000118
    sin3(x) =  sin(0.002*pi*i*(1000/(5000*17))); %0.0000074
    sin4(x) =  sin(0.002*pi*i*(1000/(5000*37))); %0.0000054
    sumsin(x) = sin2(x) + sin3(x)+ sin4(x);
    time(x) = i;
    x = x+1;
end

%normalize
mval = max(sumsin);

normsumsin = (sumsin./mval.*a) + basetimedif;

figure;
plot(time, sin2)
figure;
plot(time, sin3)
figure;
plot(time, sin4)
figure
plot(time, sumsin)
figure
plot(time, normsumsin)


%% decreasing chirp with for loop
clear all;
tic;
Baseline = 80;
basetimeDif = 1000/(Baseline/60);
c = 30;
a = basetimeDif*0.2; % Amplitude
xe = 1;
endTime = 400000;

%while toc < endtime
i = 1;
delFreq = basetimeDif;
previousMillis = 0;
for i = 1:endTime
%if (currentMillis - previousMillis >= delFreq)
    %first number in brackets somehow equals the number of peaks
    %but just if the 2 inside the exp stays 2
    delFreq(i) = a*0.999992^i*sin(i*pi*0.000025)+ basetimeDif;   
    previousMillis = i;
%end 
end
figure
plot(delFreq)

%% decreasing chirp with time
clear all;
tic;
Baseline = 80;
basetimeDif = 1000/(Baseline/60);
c = 30;
a = basetimeDif*0.2; % Amplitude
xe = 1;
endTime = 60000;

%while toc < endtime
i = 1;
delFreq = basetimeDif;
previousMillis = 0;

while toc*1000 < endTime
    currentMillis = toc * 1000;
%if (currentMillis - previousMillis >= delFreq)
    %first number in brackets somehow equals the number of peaks
    %but just if the 2 inside the exp stays 2
    delFreq(i) = a*0.99999993^currentMillis*sin(2*(currentMillis*0.0005))+ basetimeDif; 
    i = i+1;
%end 
end

figure
plot(delFreq)

%% Original frequency increasing chirp (loop it)
clear all;
tic;
Baseline = 86;
basetimeDif = 500/(Baseline/60);
c = 30;
a = 0.1*basetimeDif; % Amplitude
xe = 1;
endTime = 400000;

%while toc < endtime
i = 1;
delFreq = basetimeDif;
previousMillis = 0;
for i = 1:endTime
%if (currentMillis - previousMillis >= delFreq)
    %first number in brackets somehow equals the number of peaks
    %but just if the 2 inside the exp stays 2
    delfreq(i) = a*sin(2*pi*exp(i/(400000/3))) + basetimeDif;      
    previousMillis = i;
%end 
end
figure
plot(delfreq)


%% sin
clear all;
tic;
tarpower=200;
a = 0.2*tarpower;
xe = 1;
endTime = 400000;

%while toc < endtime
i = 1;
previousMillis = 0;
for i = 1:endTime
%if (currentMillis - previousMillis >= delFreq)
    %first number in brackets somehow equals the number of peaks
    %but just if the 2 inside the exp stays 2
    power(i) = a*sin(i*pi*0.00003)+ tarpower;   
    previousMillis = i;
%end 
end
figure
plot(power)

