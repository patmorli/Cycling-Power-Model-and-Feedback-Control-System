%% nonlinear regression model, find one unknown variable
%% normalized data
%% Nov 1, 2018
clear;
close all

%% load a file
filename = 'StepOD075_v2';
load(filename);

%% define variables
GR = 0.378;
c1 = 1; %Drag
m = 84;

%% filter gps data
windowSize = 25; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;

fs = 3;
v = lowpass(meantrialspeed, 0.2, fs);
[b,a] = butter(4, .1*2/fs);
v = filtfilt(b,a,meantrialspeed);
dv = nanmean([diff([v NaN]); diff([NaN v])]);
dt = nanmean([diff([time NaN]); diff([NaN time])])/1000;
vdot = dv./dt;

figure(100); clf; hold on
    plot(meantrialspeed,'r');
    plot(v,'b')
figure(101); clf; hold on
    plot(vdot);

% v = meantrialspeed;

t=1:length(v);

%%save stuff
P=meantrialpower;
save('nonlindata', 'v', 'P', 'GR', 'c1', 'm');
clear all;



%% define all variables
% P(t)...mechanical power measured in crank
% v(t)...speed of the bike
% dv(t)...acceleration of the bike
% GR...gear ratio
% m...mass of biker plus bike
% c1...constant of drag forces

% syms v(t) t P1(t) P2(t)
% syms GR m c1 dv
% 
% %% equation
% P1 = (GR * m * dv .* v); %initial term (mainly for the bump)
% P2 =  (GR * c1 * v.^3);  %drag term (mainly for steady state) 
% P = P1 + P2;

%% load and prepare data
load('nonlindata');

%% regression start 

modelfun = @(c,v)GR*m*diff(v).*v(1:end-1) + GR*c(1)*v(1:end-1).^3; 

%v=v(1:end-1);
P=P(1:end-1);
% c(1) and c(2) are gear ratio and drag respectively
% and also the variables we want to estimate
c = [c1];

%% regression calculations
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
[result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v,P,modelfun,c, opts);
c1new = result;

%% fill in calculated variable 
Pnew = GR*m*diff(v).*v(1:end-1) + GR*c1new*v(1:end-1).^3; 

%% plot
figure(1);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Comparison Derived Power and Actual Power')
xlabel('Trialtime [normalized]') 
ylabel('Power [W]') 
hold on;
plot(P, 'b', 'LineWidth', 3);
plot(Pnew, 'r', 'LineWidth', 3);
legend({'Measured','Simulated'}, 'FontSize', 16)



