clear
close all

%% load your data
% fp = '/Volumes/GoogleDrive/My Drive/Cycling Power Control/Work/OpenLoop/Matlab';
% fn = 'Ch05OD_pat_v1.mat';
% load(fullfile(fp,fn));
load("PROTPat_v1.mat");

%% plot your data 
figure(1); clf; hold on
plot(time, bpm); plot(time, rpm);
%%
rpmcut = rpm(100:1000);
bpmcut = bpm(100:1000);

%% make id data with input and output
data = iddata(bpmcut',rpmcut',1);

%% do calculations. Output is in sys and sys2 variable. Probably
%% all the variables that you need in there. FPE = Final Prediction Error
%% Td = Timedelay, Tp = timeconstant (time of the step response to reach 2/3 of the final value)
%% Kp = gain
opt = procestOptions;
opt.Display = 'on';
opt.SearchOption.MaxIter = 100;
sys = procest(data,'P1D',opt)
%sys = tfest(data,np);

%%
figure
compare(data,sys);

%%
sys2 = pem(data,sys)

