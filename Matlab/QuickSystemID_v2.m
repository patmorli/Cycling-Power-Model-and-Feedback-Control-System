function [sys, sys2,data] = QuickSystemID_v2(inputtrialsNorm,outputtrialsNorm, numIt)

%%
%clear
%close all
%load(fn);

%%
% fp = '/Volumes/GoogleDrive/My Drive/Locomotion Lab/Projects/Cycling Power Control/Work/OpenLoop/Matlab';
% fn = 'Patrick_Openloop_v6.mat';
% load(fullfile(fp,fn));

%%
% fp = '/Volumes/GoogleDrive/My Drive/Cycling Power Control/Work/OpenLoop/Matlab';
% fn = 'Patrick_Openloop_v6.mat';
% load(fullfile(fp,fn));



%%
figure(1); clf; hold on
plot(mean(outputtrialsNorm))

%%
for i = 1:size(outputtrialsNorm,1)
    line = iddata(outputtrialsNorm(i,:)',inputtrialsNorm(i,:)',1);
    eval(['data' num2str(i) '= line']);
    if i == 1
        together = 'data1';
    else
        together = strcat(together, ',', ['data' num2str(i)]);
    end 
end

%% I have to find a dynamical way for that... 
data = merge(data1, data2, data3, data4, data5, data6, data7, data8, data9,...
    data10, data11, data12, data13, data14, data15, data16, data17, ...
    data18, data19, data20, data21, data22);
%data = merge(eval(together)); %this unfortunately does not work, no idea
%why

% data.TimeUnit = 'cycles';
% data = iddata(powertrialsNorm',rpmtrialsNorm',1)

%%
opt = procestOptions;
opt.Display = 'on';
opt.SearchOption.MaxIter = numIt;
sys = procest(data,'P1D',opt)
%sys = tfest(data,1);
%%
figure
compare(data,sys);
hold on;
plot(mean(outputtrialsNorm),'r');
plot(mean(inputtrialsNorm), '--b');

%%
sys2 = pem(data,sys)
%figure
%compare(data,sys2);
end




