
%%
clear
close all

%%
% fp = '/Volumes/GoogleDrive/My Drive/Cycling Power Control/Work/OpenLoop/Matlab';
% fn = 'Patrick_Openloop_v7.mat';
% load(fullfile(fp,fn));
load("Patrick_Openloop_v7.mat");

%%
% fp = '/Volumes/GoogleDrive/My Drive/Cycling Power Control/Work/OpenLoop/Matlab';
% fn = 'Patrick_Openloop_v7.mat';
% load(fullfile(fp,fn));

%%
figure(1); clf; hold on
plot(mean(powertrialsNorm))

%%
data1 = iddata(powertrialsNorm(1,:)',bpmtrialsNorm(1,:)',1);
data2 = iddata(powertrialsNorm(2,:)',bpmtrialsNorm(2,:)',1);
data3 = iddata(powertrialsNorm(3,:)',bpmtrialsNorm(3,:)',1);
data4 = iddata(powertrialsNorm(4,:)',bpmtrialsNorm(4,:)',1);
data5 = iddata(powertrialsNorm(5,:)',bpmtrialsNorm(5,:)',1);
data6 = iddata(powertrialsNorm(6,:)',bpmtrialsNorm(6,:)',1);
data7 = iddata(powertrialsNorm(7,:)',bpmtrialsNorm(7,:)',1);
data = merge(data1,data2,data4,data5,data6,data7)
% data.TimeUnit = 'cycles';
% data = iddata(powertrialsNorm',rpmtrialsNorm',1)

%%
opt = procestOptions;
opt.Display = 'on';
opt.SearchOption.MaxIter = 100;
sys = procest(data1,'P1D',opt)
%sys = tfest(data,np);
%%
figure
compare(data,sys);

%%
sys2 = pem(data,sys)






