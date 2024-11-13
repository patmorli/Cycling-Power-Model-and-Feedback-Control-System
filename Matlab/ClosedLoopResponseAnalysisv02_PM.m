%%
% The purpose of this code is to simulate the behaviour of the speed
% control controller to better choose controller parameters.
%
% I have a report on this called Theoretical Controller Performance Report.
clear
close all

%% NOTES
% - The most striking thing is that the measurement noise and delay don't
% matter much if we keep the gain low. It suggests that much of the issue
% is that people are not matching the tempo. 
% - Decreasing Ki to handle the large slopes for some people, makes for
% very slow responses for those with shallow slopes. 
% - How do we know what the acceptable speed fluctuations are?
% - With small Ki, gps noise and even gps going to fast for a while and too
% slow for a while doesn't seem to lead to fluctuations that are too large.
% - It doesn't seem like measurement delay matters that much. 


%%
% Specifically, the fast and slow processes underlying speed selection had
% respective time constants of 1.9±0.8s and 30.5±6.4 second (p=2.4x10-4,
% Wilcoxon signed-rank test). The time delay prior to responding to
% perturbations was 1.6±1.7s. While increases in prescribed step frequency
% consistently resulted in increases to steady-state running speed, the
% slope of the relationship varied considerably between participants. It
% ranged from 1.4 to 3.3 ms-1Hz-1, with an average slope of 2.3±0.7
% ms-1Hz-1 (Mark's paper). The resulting default proportional and integral
% gain used for the control of running speed was 0.06 and 0.05,
% respectively (Mark's dissertation - vBox controller). The individual
% controller settings for each participant are given in table B-1. Fast
% time constant range 0.8 - 3.8; % eyeballing Mark's bar plot of individual
% subjects.

%% Block Parameters

% Plant Dynamics Longer time constant, larger slope, and longer time delay
% are all worse for stability. It is most sensitive to the slope. This is
% the mathematically same effect as the integral gain.
tFast = 1.9+2*0.8; % in seconds; mean is 1.9±0.8s; eyeballed range is 0.8-3.8; 
runnerDelay = 1.6+2*1.7; % in seconds; mean is 1.6±1.7s; I don't know the range.
SpeedFreqSlope = 2.3+2*0.7;  % in (m/s)/Hz; mean is 2.3±0.7; range is 1.4 to 3.3
% % average - used the worst case above for robustness.
tFast = 1.9; % in seconds; mean is 1.9±0.8s; eyeballed range is 0.8-3.8; 
runnerDelay = 1.6; % in seconds; mean is 1.6±1.7s; I don't know the range. 
SpeedFreqSlope = 2.3;  % in (m/s)/Hz; mean is 2.3±0.7; range is 1.4 to 3.3

measurementDelay = 3; % in seconds; measured as 2 seconds for gpsSpeed. See report. 
% measurementNoiseVariance = 1*[(0.168)^2]; % in (m/s)^2; std measured as 0.168 m/s and not signal dependent. See report.
measurementNoiseVariance = 1*[(0.5)^2]; % in (m/s)^2; std measured as 0.50 m/s in some of Ariel's runs. See report.
measurementNoiseVariance = 1*[(0.3)^2]; % in (m/s)^2; std measured as 0.50 m/s in some of Ariel's runs. See report.
measurementNoiseSeed = 0; % seeds the random number generator. 
measurementOffsetGenPeriod = 20*60;

% Integral gain. 
Ki = 0.01; % units of steps per second per metre; vBox controller used 0.05; CC app used 0.03; We have been testing with 0.005. 

% Desired speed. This is really a step change from 0 (perhaps to 0 as well
% to study state). This is convenient so we don't need to worry about
% initial conditions.
% Step Change
sDesired = 0.5; % in m/s
offsetSpeed = 2.5; % in m/s
% Steady State
sDesired = 0.0; % in m/s
offsetSpeed = 1000*(1/5.5)/60; % in m/s

% Build runner transfer function. 
num = 1;
den = [tFast 1];
sys = tf(num,den)
sysd = c2d(sys,1)
num = sysd.Numerator{1}
denom = sysd.Denominator{1}
%%
paramNameValStruct.SimulationMode = 'normal';
paramNameValStruct.StopTime = num2str(10*60);
paramNameValStruct.AbsTol         = '1e-5';
paramNameValStruct.SaveState      = 'on';
paramNameValStruct.StateSaveName  = 'xout';
paramNameValStruct.SaveOutput     = 'on';
paramNameValStruct.OutputSaveName = 'yout';
% 'SaveFormat', 'Dataset'

%%
mdlName = 'ClosedLoopSpeedControlSimulinkModelv03_PM';
set_param([mdlName '/speedDesired'],'Value',num2str(sDesired))
set_param([mdlName '/runnerDelay'],'DelayLength',num2str(round(runnerDelay)))
set_param([mdlName '/SpeedFreqSlope'],'Gain',num2str(SpeedFreqSlope))
set_param([mdlName '/measurementDelay'],'DelayLength',num2str(round(measurementDelay)))
set_param([mdlName '/measurementNoise'],'Seed',num2str(measurementNoiseSeed))
set_param([mdlName '/measurementNoise'],'Variance',num2str(measurementNoiseVariance))
set_param([mdlName '/measurementOffsetGen'],'Period',num2str(measurementOffsetGenPeriod))
set_param([mdlName '/Ki'],'Gain',num2str(Ki))
set_param([mdlName '/runnerTF'],'Numerator',['[' num2str(num(1)) ' ' num2str(num(2)) ']'])
set_param([mdlName '/runnerTF'],'Denominator',['[' num2str(denom(1)) ' ' num2str(denom(2)) ']'])

%%
simOut = sim('ClosedLoopSpeedControlSimulinkModelv03_PM',paramNameValStruct);
outputs = simOut.get('yout');
time=(outputs.get('actualSpeed').Values.Time);
desiredSpeed=(outputs.get('desiredSpeed').Values.Data);
actualSpeed=(outputs.get('actualSpeed').Values.Data);
measuredSpeed=(outputs.get('measuredSpeed').Values.Data);
stepFrequency=(outputs.get('stepFrequency').Values.Data);
beforeDelay=(outputs.get('beforeDelay').Values.Data);
afterDelay=(outputs.get('afterDelay').Values.Data);
Noise=(outputs.get('Noise').Values.Data);
Offset=(outputs.get('Offset').Values.Data);


%%
S = stepinfo(actualSpeed,time,sDesired,'SettlingTimeThreshold',0.1)
figure(1); hold on
    plot(time,actualSpeed)
    box on
    grid on
    
%%
desiredPace = (1./(offsetSpeed+desiredSpeed).*1000/60);
actualPace = (1./(offsetSpeed+actualSpeed).*1000/60);
measuredPace = (1./(offsetSpeed+measuredSpeed).*1000/60);

cadenceChange = stepFrequency*60;

%% Steady State Metrics
% These make the most sense when the sDesired is 0 so that we are studying
% things in steady state.
[mean(actualPace) std(actualPace) 100*std(actualPace)./mean(actualPace) 100*range(actualPace./mean(actualPace))]


%%
t = time;
% time = time/60;

%%
if 1
figure(2); clf
    set(gcf,'Position',[0 0 900 800])
    subplot(211); hold on
        plot(time,desiredPace,'g','linewidth',[1.5])
        plot(time,actualPace,'b','linewidth',[1])
        plot(time,measuredPace,'color',[1 0 0 0.2])
        plot(time,5.25*ones(length(time)),'k')
        plot(time,5.75*ones(length(time)),'k')
%         plot(time,measuredPace)
%         ylim([5.25 5.75])
%         xlim([0 10])
%         ylim([3.5 7.5])
        box on 
        grid on
        ylabel('Pace (min/km)')
        legend({'desiredPace','actualPace','measuredPace'},'Location','northeast')
%         title(['Ki: ' num2str(Ki,4) ';  Actual Pace Standard Deviation: ' num2str(100*std(actualPace)./mean(actualPace),2) '%'])
        title(['Ki: ' num2str(Ki,4) ';  Actual Pace Standard Deviation: ' num2str(100*std(actualPace)./mean(actualPace),2) '%'])

    subplot(212); hold on
        plot(time,cadenceChange,'b','linewidth',[1])
        box on 
        grid on
        ylabel('Cadence Change (steps per minute)')
        plot(time,2*ones(length(time)),'k')
        plot(time,-2*ones(length(time)),'k')
%         ylim([-5 5])

        xlabel('Time (minutes)')

end
linkaxes(get(gcf,'children'),'x')
%%
if 0 % used this for paremeter sweeps.
P = [0.005 0.0075 0.01 0.02]; % Ki
P = [1:10]; % runnerDelay
P = [2.3-2*0.7 2.3 2.3+2*0.7]; % SpeedFreqSlope.
P = [0.5 1 1.5 2 5 10]; % tFast
P = [1 2 5 10 15 20]; % measurementDelay
% P = [0.25 0.5 1.0 1.5 2.0 3.0]; % measurementNoise
    

P = [2 10 50 100]; % measurementDelay

clear S
figure(1); clf; hold on
for i=1:length(P)
%     Ki(i) = P(i);
%     runnerDelay = P(i);
%     SpeedFreqSlope = P(i);
%     tFast = P(i);    
%     num = 1;
%     den = [tFast 1];
%     sys = tf(num,den)
%     sysd = c2d(sys,1)
%     num = sysd.Numerator{1}
%     denom = sysd.Denominator{1}

%     measurementDelay = P(i);
%     measurementNoiseVariance = (0.5*P(i))^2;
    measurementOffsetGenPeriod = P(i);

%     set_param([mdlName '/Ki'],'Gain',num2str(Ki(i)))
%     set_param([mdlName '/speedDesired'],'Value',num2str(sDesired))
%     set_param([mdlName '/runnerDelay'],'DelayLength',num2str(round(runnerDelay)))
%     set_param([mdlName '/measurementDelay'],'DelayLength',num2str(round(measurementDelay)))
%     set_param([mdlName '/measurementNoise'],'Seed',num2str(measurementNoiseSeed))
%     set_param([mdlName '/measurementNoise'],'Variance',num2str(measurementNoiseVariance))
set_param([mdlName '/measurementOffsetGen'],'Period',num2str(measurementOffsetGenPeriod))
%     set_param([mdlName '/SpeedFreqSlope'],'Gain',num2str(SpeedFreqSlope))
%     set_param([mdlName '/Ki'],'Gain',num2str(Ki))
%     set_param([mdlName '/runnerTF'],'Numerator',['[' num2str(num(1)) ' ' num2str(num(2)) ']'])
%     set_param([mdlName '/runnerTF'],'Denominator',['[' num2str(denom(1)) ' ' num2str(denom(2)) ']'])
    simOut = sim('ClosedLoopSpeedControlSimulinkModelv03',paramNameValStruct);
    outputs = simOut.get('yout');
    time=(outputs.get('actualSpeed').Values.Time);
    desiredSpeed=(outputs.get('desiredSpeed').Values.Data);
    actualSpeed=(outputs.get('actualSpeed').Values.Data);
    measuredSpeed=(outputs.get('measuredSpeed').Values.Data);
    stepFrequency=(outputs.get('stepFrequency').Values.Data);
    
    var(i) = std(actualSpeed(100:end));
    
    
    S(i) = stepinfo(actualSpeed,time,sDesired,'SettlingTimeThreshold',0.1,'RiseTimeLimits',[0 0.9])
    plot(time,actualSpeed+offsetSpeed)
    pause(0.01)
    ylim([2 3.5])
    box on
    grid on
    ylabel('Speed (m/s)')
    xlabel('Time (s)')
end
xl = get(gca,'xlim');
plot(xl,1.05*3*[1 1],'k')
plot(xl,0.95*3*[1 1],'k')
legend(num2str(P',2))
end

