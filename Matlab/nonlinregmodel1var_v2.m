% nonlinear regression model, find one unknown variable
% with data of whole trial
% Nov 13, 2018

clear all;
close all

%% load a file
filename = 'StepOD075_v2';
load(filename);

%% define variables
GR = 0.378;
c1 = 1; %Drag
m = 84;
t = (time/1000)';

%% filter data
fs = 3;
[b,a] = butter(4, .1*2/fs);
v = filtfilt(b,a,gpsSpeed)';
P = filtfilt(b,a,power);

%% derive
dv = nanmean([diff([v NaN]); diff([NaN v])]); % central difference
dt = nanmean([diff([t NaN]); diff([NaN t])]);
vdot = dv./dt;


%% regression start 
modelfun = @(c,v)GR*m*vdot.*v + GR*c(1)*v.^3; 
c = [c1];


%% regression calculations
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
[result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v,P,modelfun,c, opts);
c1new = result;

%% fill in calculated variable 
Pnew = GR*m*vdot.*v + GR*c1new*v.^3; 

%% plot
figure(1);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Comparison Derived Power and Actual Power')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(t, P, 'b', 'LineWidth', 3);
plot(t, Pnew, 'r', 'LineWidth', 3);
legend({'Measured','Simulated'}, 'FontSize', 16)
