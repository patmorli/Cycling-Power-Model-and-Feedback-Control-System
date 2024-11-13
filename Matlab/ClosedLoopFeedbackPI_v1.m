%% ClosedLoopFeedback Proportional and Integral Gain of Power with Cadence
% Started May, 2019, Patrick Mayerhofer
% Tries to model a feedback system with the model of
% the biker's dynamics and a controller
clear all;
close all;
open_system('ClosedLoopFedbackSimPI_v1');
%% some constants
rpm = 86; %starting rpm
fbracket = 39;
rbracket = 19;
GR = fbracket/rbracket;
rw=0.3344; %radius wheel
lca = 0.17; %length crank arm
gearing = (rw/lca)*GR;
m = 74;
const = 60;
AmpOffset = 236;
ChirpMulti = 50;
Ki = 0.02;
Kp = 0.012;
c1 = 0.36;
rpmNoise = 25;
rpmSeed = 2;
rpmNoise1 = 50;
rpmSeed1 = 0;
measurementSeed = 3;
measurementNoise = 5;
bikerDelay = 0;

Amplitude = 18;
Period = 600;
Width = 300;
Frequency = 50;
%Chirp
Initialfrequency = 0.0001;
Targetfrequency = 0.1;
measurementDelay = 0;

%% 
paramNameValStruct.SimulationMode = 'normal';
paramNameValStruct.StopTime = num2str(10*120);
paramNameValStruct.AbsTol         = '1e-5';
paramNameValStruct.SaveState      = 'on';
paramNameValStruct.StateSaveName  = 'xout';
paramNameValStruct.SaveOutput     = 'on';
paramNameValStruct.OutputSaveName = 'yout';

%% feed into simulink model
mdlName = 'ClosedLoopFedbackSimPI_v1';
set_param([mdlName '/lca'],'Value',num2str(lca))
set_param([mdlName '/rw'],'Value',num2str(rw))
set_param([mdlName '/pi'],'Value',num2str(pi))
set_param([mdlName '/GR'],'Value',num2str(GR))
set_param([mdlName '/const'],'Value',num2str(const))
set_param([mdlName '/m'],'Value',num2str(m))
set_param([mdlName '/c1'],'Value',num2str(c1))
set_param([mdlName '/AmpOffset'],'Value',num2str(AmpOffset))
set_param([mdlName '/ChirpMulti'],'Value',num2str(ChirpMulti))
set_param([mdlName '/Ki'],'Gain',num2str(Ki))
set_param([mdlName '/Kp'],'Gain',num2str(Kp))
set_param([mdlName '/bikerDelay'],'DelayLength',num2str(round(bikerDelay)))
set_param([mdlName '/measurementDelay'],'DelayLength',num2str(round(measurementDelay)))


%Noise
set_param([mdlName '/rpmNoise'],'Seed',num2str(rpmSeed))
set_param([mdlName '/rpmNoise'],'Variance',num2str(rpmNoise))
set_param([mdlName '/rpmNoise1'],'Seed',num2str(rpmSeed1))
set_param([mdlName '/rpmNoise1'],'Variance',num2str(rpmNoise1))
set_param([mdlName '/measurementNoise'],'Seed',num2str(measurementSeed))
set_param([mdlName '/measurementNoise'],'Variance',num2str(measurementNoise))
%StepFunction
set_param([mdlName '/Pulse'],'Amplitude',num2str(Amplitude))
set_param([mdlName '/Pulse'],'Period',num2str(Period))
set_param([mdlName '/Pulse'],'PulseWidth',num2str(Width))


%Chirp
% set_param([mdlName '/Chirp'],'InitialFrequency',num2str(Initialfrequency))
% set_param([mdlName '/Chirp'],'TargetFrequency',num2str(Targetfrequency))

%Sine Wave Function
% set_param([mdlName '/Sine'],'Amplitude',num2str(Amplitude))
% set_param([mdlName '/Sine'],'Frequency',num2str(Frequency))


    simOut = sim('ClosedLoopFedbackSimPI_v1',paramNameValStruct);
    outputs = simOut.get('yout');
    time =(outputs.get('desiredPower').Values.Time);
    desiredPower=(outputs.get('desiredPower').Values.Data);
    outputPower=(outputs.get('outputPower').Values.Data);
    measuredPower=(outputs.get('measuredPower').Values.Data);
    threshold = (outputs.get('threshold').Values.Data);
    bpmsat=(outputs.get('bpmsat').Values.Data);
    bpmbefsat=(outputs.get('bpmbefsat').Values.Data);
    speed=(outputs.get('speed').Values.Data);
    error=(outputs.get('error').Values.Data);
    integrator=(outputs.get('integrator').Values.Data);
    Kigain=(outputs.get('Kigain').Values.Data);
    Kpgain=(outputs.get('Kpgain').Values.Data);
    rpmout=(outputs.get('rpm').Values.Data);
    Noise=(outputs.get('Noise').Values.Data);
    Noise1=(outputs.get('Noise1').Values.Data);
    SineOut=(outputs.get('SineOut').Values.Data);
    targetSignal=(outputs.get('targetSignal').Values.Data);
    acc=(outputs.get('acc').Values.Data);
    rpmdot=(outputs.get('rpmdot').Values.Data);


% for i = 1:length(Ki_vals)
%     simIn(i) = Simulink.SimulationInput(mdlName);
%     simIn(i) = setVariable(simIn(i),'Ki',Ki_vals(i));
% end


%% get out of simulink model

% outputs = simOut.get('yout');
% time=(outputs.get('desiredPower').Values.Time);
% desiredPower=(outputs.get('desiredPower').Values.Data);
% outputPower=(outputs.get('outputPower').Values.Data);
% measuredPower=(outputs.get('measuredPower').Values.Data);
%  bpm=(outputs.get('bpm').Values.Data);
% speed=(outputs.get('speed').Values.Data);
% error=(outputs.get('error').Values.Data);
% integrator=(outputs.get('integrator').Values.Data);
% Kigain=(outputs.get('Kigain').Values.Data);
% Kpgain=(outputs.get('Kpgain').Values.Data);
% bpm=(outputs.get('bpm').Values.Data);
% rpm=(outputs.get('rpm').Values.Data);
% Noise=(outputs.get('Noise').Values.Data);
% Noise1=(outputs.get('Noise1').Values.Data);
% SineOut=(outputs.get('SineOut').Values.Data);
% targetSignal=(outputs.get('targetSignal').Values.Data);

MSE=immse(measuredPower, desiredPower);

fs = 3;
% fs = 10;
[b,a] = butter(4, .3*2/fs);
measuredPowerfilt = filtfilt(b,a,measuredPower());
rpmfilt = filtfilt(b,a,rpmout());
time2 = 0;
% get time of datapoints
for i=1:length(rpmfilt)
        time2(i+1)=time2(i)+1/(rpmfilt(i)/60);
    
    %rpm(i) = 30000/(time(i+1)-time(i));

end

%% plots 
figure;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
xlabel('Trialtime [s]');
yyaxis left; hold on; plot(time, bpmsat, 'b', 'LineWidth', 3); ylabel('Cadence [bpm]'); 
yyaxis right; plot(time, measuredPower, 'r', 'LineWidth', 3); ylabel('Power [W]') ; 
plot(time, desiredPower, '--g', 'LineWidth', 3); 
legend({'Cadence','measuredPower', 'desiredPower'}, 'FontSize', 16);
title('Feedback Controller Results, Ki=0.02, Kp=0.01');


figure(4);
subplot(2,1,1);
hold on;
% ax = gca;
% ax.LabelFontSizeMultiplier = 2;
% ax.TitleFontSizeMultiplier = 2;
title('Cadence', 'fontweight', 'bold')
plot(time(300:end)-time(300), rpmout(300:end)); 
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Cadence [rpm]', 'fontweight', 'bold');
hold off;

subplot(2,1,2); 
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power', 'fontweight', 'bold')
plot(time(300:end)-time(300), measuredPower(300:end)); plot(time(300:end)-time(300),desiredPower(300:end));
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');
hold off;



