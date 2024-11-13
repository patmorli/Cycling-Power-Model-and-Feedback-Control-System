%% ClosedLoopFeedback of Power with Cadence
% Started May, 2019, Patrick Mayerhofer
% Tries to model a feedback system with the model of
% the biker's dynamics and a controller
clear all;
close all;
%% some constants
rpm = 86; %starting rpm
fbracket = 39;
rbracket = 21;
GR = fbracket/rbracket;
rw=0.3344; %radius wheel
lca = 0.17; %length crank arm
gearing = (rw/lca)*GR;
m = 74;
v = (rw * 2*pi*GR.*rpm)/60;
const = 60;
desiredPower = 200;
Ki = 0.01;
c1 = 0.3;

%% 
paramNameValStruct.SimulationMode = 'normal';
paramNameValStruct.StopTime = num2str(10*60);
paramNameValStruct.AbsTol         = '1e-5';
paramNameValStruct.SaveState      = 'on';
paramNameValStruct.StateSaveName  = 'xout';
paramNameValStruct.SaveOutput     = 'on';
paramNameValStruct.OutputSaveName = 'yout';

%% feed into simulink model
mdlName = 'ClosedLoopFedbackSim_v1';
set_param([mdlName '/lca'],'Value',num2str(lca))
set_param([mdlName '/rw'],'Value',num2str(rw))
set_param([mdlName '/pi'],'Value',num2str(pi))
set_param([mdlName '/GR'],'Value',num2str(GR))
set_param([mdlName '/const'],'Value',num2str(const))
set_param([mdlName '/m'],'Value',num2str(m))
set_param([mdlName '/c1'],'Value',num2str(c1))
set_param([mdlName '/desiredPower'],'Value',num2str(desiredPower))
set_param([mdlName '/Ki'],'Gain',num2str(Ki))

%% get out of simulink model
simOut = sim('ClosedLoopFedbackSim_v1',paramNameValStruct);
outputs = simOut.get('yout');
time=(outputs.get('actualPower').Values.Time);
desiredPower=(outputs.get('desiredPower').Values.Data);
actualPower=(outputs.get('actualPower').Values.Data);
bpm=(outputs.get('bpm').Values.Data);
speed=(outputs.get('speed').Values.Data);
error=(outputs.get('error').Values.Data);
integrator=(outputs.get('integrator').Values.Data);


figure; plot(actualPower);
figure; plot(bpm);
