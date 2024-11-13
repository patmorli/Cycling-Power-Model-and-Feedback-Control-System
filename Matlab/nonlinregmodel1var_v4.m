% nonlinear regression model, find one unknown variable
% with data of whole trial
% Nov 13, 2018

clear all;
close all

%% load a file
filename = 'Subject1_1';
load(filename);

%% define variables

fbracket = 39;
rbracket = 27;
GR = fbracket/rbracket;

rw=0.3; %radius wheel
lca = 0.17; %length crank arm
m = 74;
m = m + 8;
t = (time/1000)';
c1 =1;

x0 = 0.9; %Drag for initial guess
%vcalc = (0.3344 * 2*pi*GR)./(1./(rpm/60));

%% data you want to optimize for
% % cut data to (100:end-100)
% rpmcut = rpm(100:end-100);
% powercut = power(100:end-100);
% tcut = t(100:end-100);

%cut data to (100:end-100)
rpmcut = rpm(30:2830);
powercut = power(30:2830);
tcut = t(30:2830);
bpmcut = bpm(30:2830);



% calculated speed from gear ratio and rpm
vcalc = (rw * 2*pi*GR.*rpmcut)/60;


%% filter data
fs = 3;
[b,a] = butter(4, .1*2/fs);
v = filtfilt(b,a,vcalc);
P = filtfilt(b,a,powercut);

%% plot data of optimization
figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power and RPM of data used for optimization')
xlabel('Trialtime [s]') 

hold on;
yyaxis left; plot(tcut, rpmcut, 'b', 'LineWidth', 3);
ylabel('Cadence [RPM]');
yyaxis right; plot(tcut, P, 'R', 'LineWidth', 3);
ylabel('Power [W]');
legend({'Cadence','Power'}, 'FontSize', 16);


%% derive
dv = nanmean([diff([v NaN]); diff([NaN v])]); % central difference
dt = nanmean([diff([tcut NaN]); diff([NaN tcut])]);
vdot = dv./dt;




%% regression start 
modelfun = @(u,v)(rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*u(1)*v.^3;
u = [c1]; %unknown

%% regression calculations
opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
 [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v,P,modelfun,u, opts);
 c1new = result(1);

 
 %% look at accuracy of models
% % look at whole data, independent from data used for model
% rpmcut2 = rpm(100:end-100);
% powercut2 = power(100:end-100);
% tcut2 = t(100:end-100);
% vcalc2 = (rw * 2*pi*GR.*rpmcut2)/60;

rpmcut2 = rpm(30:2830);
powercut2 = power(30:2830);
tcut2 = t(30:2830);
vcalc2 = (rw * 2*pi*GR.*rpmcut2)/60;


%  varnew = 0.29;
%  c1new = 0.1274;
% filter
fs = 3;
[b,a] = butter(4, .1*2/fs);
v2 = filtfilt(b,a,vcalc2);
P2 = filtfilt(b,a,powercut2);
dv2 = nanmean([diff([v2 NaN]); diff([NaN v2])]); % central difference
dt2 = nanmean([diff([tcut2 NaN]); diff([NaN tcut2])]);
vdot2 = dv2./dt2;
 %% fill in calculated variable and calcu
 Pnewreg = (rw/lca)*GR*m*vdot2.*v2 + (rw/lca)*GR*c1new*v2.^3; 
 
 
 
 %% variance
 [rsquaredreg,msereg, rmsereg] = cost(P2,Pnewreg);


figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Comparison Derived Power and Actual Power-reg')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(tcut2, Pnewreg, 'r', 'LineWidth', 3);
plot(tcut2, P2, 'b', 'LineWidth', 3);
legend({'Simulated','Measured'}, 'FontSize', 16);


%save(filename);