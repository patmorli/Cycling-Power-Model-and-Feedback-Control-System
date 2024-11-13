%% nonlinear regression model, find two unknown variables 
%% normalized data
%% Jan 29, 2019
clear all;
close all;

%% load a file
filename = 'StepOD05_v2';
load(filename);

%% define variables
%GR = 0.378;
fbracket = 39;
rbracket = 27;
rw=0.3344; %radius wheel
lca = 0.17; %length crank arm

GR = fbracket/rbracket;
c1 = 0.1; %Drag for initial guess
m = 84;
t = (time/1000)';
%vcalc = (0.3344 * 2*pi*GR)./(1./(rpm/60));
vcalc = (rw * 2*pi*GR.*rpm)/60;

%% prepare speed and acceleration

%% filter data
fs = 3;
[b,a] = butter(4, .1*2/fs);
v = filtfilt(b,a,vcalc);
P = filtfilt(b,a,power);

%% derive
dv = nanmean([diff([v NaN]); diff([NaN v])]); % central difference
dt = nanmean([diff([t NaN]); diff([NaN t])]);
vdot = dv./dt;

%% regression start 
modelfun = @(u,v)(rw/lca)*GR*u(2)*vdot.*v + (rw/lca)*GR*u(1)*v.^3;
u = [c1 m]; %unknown

%% regression calculations
%opts = statset('nlinfit');

opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
opts.Display = 'iter';
[result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v,P,modelfun,u, opts);
c1new = result(1);
mnew = result(2);

%% fill in calculated variable 
Pnew = (rw/lca)*GR*mnew*vdot.*v + (rw/lca)*GR*c1new*v.^3; 

%% variance
[r, P1]=corrcoef(P, Pnew);
r2 = r(2,1)^2;

%% plot
figure(1);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Comparison Derived Power and Actual Power')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(P, 'b', 'LineWidth', 3);
plot(Pnew, 'r', 'LineWidth', 3);
legend({'Measured','Simulated'}, 'FontSize', 16);
save('atwo');
