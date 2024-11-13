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
AmpOffset = 100;
ChirpMulti = 50;
% Ki = 0.1;
% Kp = 1;
c1 = 0.3;
rpmNoise = 25;
rpmSeed = 2;
rpmNoise1 = 50;
rpmSeed1 = 0;
measurementSeed = 3;
measurementNoise = 5;

Amplitude = 100;
Period = 400;
Width = 200;
Frequency = 50;
%Chirp
Initialfrequency = 0.0001;
Targetfrequency = 0.1;
bikerDelay = 0;
measurementDelay = 0;

%% 
paramNameValStruct.SimulationMode = 'normal';
paramNameValStruct.StopTime = num2str(10*60);
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
%set_param([mdlName '/Ki'],'Gain',num2str(Ki))
%set_param([mdlName '/Kp'],'Gain',num2str(Kp))
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

Ki_vals = 0:0.5:2;
Kp_vals = 0:0.5:2;
%%
for i = 1:length(Ki_vals)
    for j = 1:length(Kp_vals)
        simIn(i,j) = Simulink.SimulationInput('ClosedLoopFedbackSimPI_v1');
        simIn(i,j) = setVariable(simIn(i),'Ki',Ki_vals(i));
        simIn(i,j) = setVariable(simIn(i),'Kp',Kp_vals(j));
        simIn(i,j) = setVariable(simIn(i),'lca',lca);
        simIn(i,j) = setVariable(simIn(i),'rw',rw);
        simIn(i,j) = setVariable(simIn(i),'pi',pi);
        simIn(i,j) = setVariable(simIn(i),'GR',GR);
        simIn(i,j) = setVariable(simIn(i),'const',const);
        simIn(i,j) = setVariable(simIn(i),'m',m);
        simIn(i,j) = setVariable(simIn(i),'c1',c1);
        simIn(i,j) = setVariable(simIn(i),'AmpOffset',AmpOffset);
        simIn(i,j) = setVariable(simIn(i),'ChirpMulti',ChirpMulti);
        simIn(i,j) = setVariable(simIn(i),'bikerDelay',bikerDelay);
        simIn(i,j) = setVariable(simIn(i),'measurementDelay',measurementDelay);
        simIn(i,j) = setVariable(simIn(i),'m',m);
        simIn(i,j) = setVariable(simIn(i),'c1',c1);
        
%        %% feed into simulink model
% mdlName = 'ClosedLoopFedbackSimPI_v1';
% set_param([mdlName '/lca'],'Value',num2str(lca))
% set_param([mdlName '/num2str(rw)'],'Value',num2str(rw))
% set_param([mdlName '/pi'],'Value',num2str(pi))
% set_param([mdlName '/GR'],'Value',num2str(GR))
% set_param([mdlName '/const'],'Value',num2str(const))
% set_param([mdlName '/m'],'Value',num2str(m))
% set_param([mdlName '/c1'],'Value',num2str(c1))
% set_param([mdlName '/AmpOffset'],'Value',num2str(AmpOffset))
% set_param([mdlName '/ChirpMulti'],'Value',num2str(ChirpMulti))
% %set_param([mdlName '/Ki'],'Gain',num2str(Ki))
% %set_param([mdlName '/Kp'],'Gain',num2str(Kp))
% set_param([mdlName '/bikerDelay'],'DelayLength',num2str(round(bikerDelay)))
% set_param([mdlName '/measurementDelay'],'DelayLength',num2str(round(measurementDelay)))
% 
% 
% %Noise
% set_param([mdlName '/rpmNoise'],'Seed',num2str(rpmSeed))
% set_param([mdlName '/rpmNoise'],'Variance',num2str(rpmNoise))
% set_param([mdlName '/rpmNoise1'],'Seed',num2str(rpmSeed1))
% set_param([mdlName '/rpmNoise1'],'Variance',num2str(rpmNoise1))
% set_param([mdlName '/measurementNoise'],'Seed',num2str(measurementSeed))
% set_param([mdlName '/measurementNoise'],'Variance',num2str(measurementNoise))
% %StepFunction
% set_param([mdlName '/Pulse'],'Amplitude',num2str(Amplitude))
% set_param([mdlName '/Pulse'],'Period',num2str(Period))
% set_param([mdlName '/Pulse'],'PulseWidth',num2str(Width))
    end
end
        simOut = sim(simIn);
        for i = 1:length(Ki_vals)
            for j = 1:length(Kp_vals)
                outputs(i,j) = simOut(i,j).get('yout');
                time(:,i, j)=(outputs(i,j).get('desiredPower').Values.Time);
                desiredPower(:,i, j)=(outputs(i,j).get('desiredPower').Values.Data);
                outputPower(:,i, j)=(outputs(i,j).get('outputPower').Values.Data);
                measuredPower(:,i, j)=(outputs(i,j).get('measuredPower').Values.Data);
                speed(:,i, j)=(outputs(i,j).get('speed').Values.Data);
                error(:,i, j)=(outputs(i,j).get('error').Values.Data);
                integrator(:,i, j)=(outputs(i,j).get('integrator').Values.Data);
                Kigain(:,i, j)=(outputs(i,j).get('Kigain').Values.Data);
                Kpgain(:,i, j)=(outputs(i,j).get('Kpgain').Values.Data);
                bpm(:,i, j)=(outputs(i,j).get('bpmbefsat').Values.Data);
                rpmout(:,i, j)=(outputs(i,j).get('rpm').Values.Data);
                Noise(:,i, j)=(outputs(i,j).get('Noise').Values.Data);
                Noise1(:,i, j)=(outputs(i,j).get('Noise1').Values.Data);
                SineOut(:,i, j)=(outputs(i,j).get('SineOut').Values.Data);
                targetSignal(:,i, j)=(outputs(i,j).get('targetSignal').Values.Data);
            end
        end
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



%% plots 
figure;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
xlabel('Trialtime [s]');
yyaxis left; hold on; plot(time, bpm, 'b', 'LineWidth', 3); ylabel('Cadence [bpm]'); 
yyaxis right; plot(time, measuredPower, 'r', 'LineWidth', 3); ylabel('Power [W]') ; 
plot(time, desiredPower, '--g', 'LineWidth', 3); 
legend({'Cadence','measuredPower', 'desiredPower'}, 'FontSize', 16);
title('Feedback Controller Results, Ki=0.01, Kp=0.01');

