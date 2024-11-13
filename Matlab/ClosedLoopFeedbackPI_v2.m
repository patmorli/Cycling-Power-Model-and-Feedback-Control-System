%% ClosedLoopFeedback Proportional and Integral Gain of Power with Cadence
% Started May, 2019, Patrick Mayerhofer
% Tries to model a feedback system with the model of
% the biker's dynamics and a controller
clear all;
close all;
mdlName = 'ClosedLoopFedbackSimPI_v1'; %'ClosedLoopFedbackSimPI_new_model', 'ClosedLoopFedbackSimPI_v1'

%% some constants
rpm = 86; %starting rpm
fbracket = 39;
rbracket = 21;
GR = fbracket/rbracket;
rw=0.33; %radius wheel
lca = 0.17; %length crank arm
gearing = (rw/lca)*GR;
m = 74;
const = 60;
noiseMean = 0; %0
noiseStd = 0; %2
AmpOffset = 236;
stopTime = 1800;
% ChirpMulti = 17;
Amplitude = 24;
Period = 1200;
Width = 600;
Frequency = 50;


Kp_vals = 0.012;
Ki_vals = 0.02;

bikerDelay = 0.9;

% filter
fs = 3;
[b,a] = butter(4, .3*2/fs);

%% specify drag number dependent on model and gear ratio
if strcmp(mdlName, 'ClosedLoopFedbackSimPI_v1')
    if rbracket == 19
        c1 = 0.3536;

    elseif rbracket == 21
        c1 = 0.3682;
    else
        disp('please specify right rbracket')
    end

elseif strcmp(mdlName, 'ClosedLoopFedbackSimPI_new_model')
    if rbracket == 19
        c1 = 1.2942;

    elseif rbracket == 21
        c1 = 1.3260;
    else
        disp('please specify right rbracket')
    end

else
    disp('specify right model')
end


%Chirp
Initialfrequency = 0.0001;
Targetfrequency = 0.1;
measurementDelay = 2;

open_system(mdlName);

%% 
paramNameValStruct.SimulationMode = 'normal';
paramNameValStruct.StopTime = num2str(stopTime);
paramNameValStruct.AbsTol         = '1e-5';
paramNameValStruct.SaveState      = 'on';
paramNameValStruct.StateSaveName  = 'xout';
paramNameValStruct.SaveOutput     = 'on';
paramNameValStruct.OutputSaveName = 'yout';

%% feed into simulink model
if strcmp(mdlName,'ClosedLoopFedbackSimPI_v1')
    set_param([mdlName '/lca'],'Value',num2str(lca))
end

set_param([mdlName '/rw'],'Value',num2str(rw))
set_param([mdlName '/pi'],'Value',num2str(pi))
set_param([mdlName '/GR'],'Value',num2str(GR))
set_param([mdlName '/const'],'Value',num2str(const))
set_param([mdlName '/m'],'Value',num2str(m))
set_param([mdlName '/c1'],'Value',num2str(c1))
set_param([mdlName '/AmpOffset'],'Value',num2str(AmpOffset))
%set_param([mdlName '/ChirpMulti'],'Value',num2str(ChirpMulti))
%set_param([mdlName '/Ki'],'Gain',num2str(Ki))
%set_param([mdlName '/Kp'],'Gain',num2str(Kp))
set_param([mdlName '/bikerDelay'],'DelayLength',num2str(round(bikerDelay)))
set_param([mdlName '/measurementDelay'],'DelayLength',num2str(round(measurementDelay)))


%Noise
set_param([mdlName '/noiseMean'],'Value',num2str(noiseMean))
set_param([mdlName '/noiseStd'],'Value',num2str(noiseStd))

% set_param([mdlName '/rpmNoise'],'Seed',num2str(rpmSeed))
% set_param([mdlName '/rpmNoise'],'Variance',num2str(rpmNoise))
% set_param([mdlName '/rpmNoise1'],'Seed',num2str(rpmSeed1))
% set_param([mdlName '/rpmNoise1'],'Variance',num2str(rpmNoise1))
% set_param([mdlName '/measurementNoise'],'Seed',num2str(measurementSeed))
% set_param([mdlName '/measurementNoise'],'Variance',num2str(measurementNoise))
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



%Kp_vals = 0.001:0.001:0.005;
%% do simulation with all different block parameter possibilities
for i = 1:length(Kp_vals)
    for j = 1:length(Ki_vals)
        set_param([mdlName '/Ki'],'Gain',num2str(Ki_vals(j)))
        set_param([mdlName '/Kp'],'Gain',num2str(Kp_vals(i)))
        if strcmp(mdlName, 'ClosedLoopFedbackSimPI_v1')
            simOut(i,j) = sim('ClosedLoopFedbackSimPI_v1',paramNameValStruct);
        elseif strcmp(mdlName, 'ClosedLoopFedbackSimPI_new_model')
            simOut(i,j) = sim('ClosedLoopFedbackSimPI_new_model',paramNameValStruct);
        else
            disp('specify right model')
        end
        
        outputs(i,j) = simOut(i,j).get('yout');
        time(:,i, j)=(outputs(i,j).get('desiredPower').Values.Time);
        desiredPower(:,i, j)=(outputs(i,j).get('desiredPower').Values.Data);
        outputPower(:,i, j)=(outputs(i,j).get('outputPower').Values.Data);
        measuredPower(:,i, j)=(outputs(i,j).get('measuredPower').Values.Data);
        dragpower(:,i,j)=(outputs(i,j).get('dragpower').Values.Data);
        inertialpower(:,i,j)=(outputs(i,j).get('inertialpower').Values.Data);
%         Abs(i,j) = (outputs(i,j).get('Abs').Values.Data);
        speed(:,i, j)=(outputs(i,j).get('speed').Values.Data);
        acceleration(:,i,j)=(outputs(i,j).get('acc').Values.Data);
        error(:,i, j)=(outputs(i,j).get('error').Values.Data);
        integrator(:,i, j)=(outputs(i,j).get('integrator').Values.Data);
        Kigain(:,i, j)=(outputs(i,j).get('Kigain').Values.Data);
        Kpgain(:,i, j)=(outputs(i,j).get('Kpgain').Values.Data);
        bpmbefsat(:,i, j)=(outputs(i,j).get('bpmbefsat').Values.Data);
        bpmaftersat(:,i, j)=(outputs(i,j).get('bpmaftersat').Values.Data);
        rpmout(:,i, j)=(outputs(i,j).get('rpm').Values.Data);
        Noise(:,i, j)=(outputs(i,j).get('Noise').Values.Data);
        Noise1(:,i, j)=(outputs(i,j).get('Noise1').Values.Data);
        SineOut(:,i, j)=(outputs(i,j).get('SineOut').Values.Data);
        targetSignal(:,i, j)=(outputs(i,j).get('targetSignal').Values.Data);
        RMSE_nofilter(i,j)=rmse(measuredPower(601:1750, i, j), desiredPower(601:1750,i,j));
        measuredPowerfilt2(:,i,j) = filtfilt(b,a,measuredPower(:,i,j));
        RMSE_filter(i,j) = rmse(measuredPowerfilt2(601:1750, i, j), desiredPower(601:1750,i,j));
        
        %acc(:,i, j)=(outputs.get('acc').Values.Data);
        %rpmdot(:,i, j)=(outputs.get('rpmdot').Values.Data);

    end
end




%% find best rmse position
% original
RMSE_nofilter_min = min(min(RMSE_nofilter));
[pos(1), pos(2)] = find(RMSE_nofilter == RMSE_nofilter_min);
optkp = Kp_vals(pos(1));
optki = Ki_vals(pos(2));

% filtered version
RMSE_filter_min = min(min(RMSE_filter));
[pos2(1), pos2(2)] = find(RMSE_filter == RMSE_filter_min);
optkp2 = Kp_vals(pos2(1));
optki2 = Ki_vals(pos2(2));


%% contour plot
% figure;
% [X1 Y1] = meshgrid(Kp_vals,Ki_vals);
% contour(X1,Y1,MSE, 'ShowText', 'On')
% hold on;
% %plot(Kp_vals(pos(1)),Ki_vals(pos(2)), 'o')



%% filter data
measuredPowerfilt = filtfilt(b,a,measuredPower(:,pos(1), pos(2)));
rpmfilt = filtfilt(b,a,rpmout(:,pos(1), pos(2)));
rpmfilt2 = filtfilt(b,a,rpmout(:,pos2(1), pos2(2)));


%% get time of datapoints
time2 = 0;
time_half_revolution = 0;
rpmfilt(1) = 1;
rpmfilt2(1) = 1;
for i=1:length(rpmfilt)
    time2(i+1)=time2(i)+1/(rpmfilt2(i)/60); % considers that we get feedback at every full revolution
    time_half_revolution(i+1)=time_half_revolution(i)+1/(rpmfilt2(i)/30); % considers that we get feedback at every half revolution
    
    %rpm(i) = 30000/(time(i+1)-time(i));

end
%% plot
figure; hold on;
plot(time_half_revolution(3:end-1), desiredPower(3:end, 1,1));
plot(time_half_revolution(3:end-1), measuredPowerfilt(3:end));



%% stats
% average absolute pacing error
meanpercentagepacingerror = mean((abs(desiredPower(1500:1700,1,1)-measuredPowerfilt(1500:1700))./desiredPower(1500:1700,1,1)));
stdpercentagepacingerror = std((abs(desiredPower(1500:1700,1,1)-measuredPowerfilt(1500:1700))./desiredPower(1500:1700,1,1)));

% coefficient of variation
cv=std(measuredPowerfilt(1500:1700)./mean(measuredPowerfilt(1500:1700)));


%% stepinfo
% get step response to start from 0
figure()
plot(time_half_revolution(1200:1750)' - time_half_revolution(1200),measuredPowerfilt(1200:1750) - measuredPowerfilt(1200));
S = stepinfo(measuredPowerfilt(1200:1750) - measuredPowerfilt(1200),time_half_revolution(1200:1750)'- time_half_revolution(1200));


%for simulation
x=time_half_revolution(1200:1750)' - time_half_revolution(1200);
y=(measuredPowerfilt(1200:1750) - min(measuredPowerfilt(1200:1750))) / (max(measuredPowerfilt(1200:1750)) - min(measuredPowerfilt(1200:1750))) - 1;

[f,g] = fit(x,y,'exp1');
timeconstant = 1/f.b;
figure;
plot(f,x,y);


%plot(S)
%[f,g] = fit(time_half_revolution(1200:1800)',measuredPowerfilt(1200:1800),'exp1');
%timeconstant = 1/f.b;
%figure;
%plot(f,time_half_revolution(1200:1800)',measuredPowerfilt(1200:1800));



%% print
S
optkp
optki
optkp2
optki2
RMSE_filter_min
RMSE_nofilter_min
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



% %% plots 
% figure;
% ax = gca;
% ax.LabelFontSizeMultiplier = 2;
% ax.TitleFontSizeMultiplier = 2;
% xlabel('Trialtime [s]');
% yyaxis left; hold on; plot(time, bpmsat, 'b', 'LineWidth', 3); ylabel('Cadence [bpm]'); 
% yyaxis right; plot(time, measuredPower, 'r', 'LineWidth', 3); ylabel('Power [W]') ; 
% plot(time, desiredPower, '--g', 'LineWidth', 3); 
% legend({'Cadence','measuredPower', 'desiredPower'}, 'FontSize', 16);
% title('Feedback Controller Results, Ki=0.01, Kp=0.01');

