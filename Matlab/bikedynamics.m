%% simulates power output with different speed and acceleration inputs
clear all;
filename = 'StepOD10_v2';
load(filename);

%% define all variables
% P(t)...mechanical power measured in crank
% v(t)...speed of the bike
% dv(t)...acceleration of the bike
% GR...gear ratio
% m...mass of biker plus bike
% c1...constant of drag forces

syms v(t) t P1(t) P2(t)
syms GR m c1 dv

%% filter gps data
windowSize = 25; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;

fs = 3;
v = lowpass(meantrialspeed, 0.2, fs);

t=1:length(v);
%v=interp1(1:length(v1),v1, linspace(1,length(v1),100),'spline');


%% equation
% P = m*diff(v) + c1*v(1:end-1).^2;
P1 = (GR * m * diff(v) .* v(1:end-1));
P2 =  (GR * c1 * v(1:end-1).^3);
P = P1 + P2;


%% define variables
GR = 0.378;
c1 = 0.73; %Drag
m = 84;

%v = 10 * sin(0.5*t)+30;
% v=Data001(:,2);
% dv = diff(v);
P1 = subs(P1);
P2 = subs(P2);
P = subs(P);

figure(1);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Comparison Derived Power and Actual Power')
xlabel('Trialtime [normalized]') 
ylabel('Power [normalized]') 
hold on;
plot(P, 'b', 'LineWidth', 3);
plot(meantrialpower, 'r', 'LineWidth', 3);
legend({'Simulated','Measured'}, 'FontSize', 16)

figure(2);
bx=gca;
bx.LabelFontSizeMultiplier = 2;
bx.TitleFontSizeMultiplier = 2;
title('Smoothened and Normalized Speed Data')
xlabel('Interpolated Trialtime [normalized]') 
ylabel('Speed [normalized]') 
hold on;
plot(v, 'b', 'LineWidth', 3);


