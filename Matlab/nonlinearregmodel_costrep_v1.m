% nonlinear regression model, cost representation 
% with data of whole trial
% Nov 13, 2018

clear all;
close all

%% load a file
filename = 'PROTPat_v3';
load(filename);

%% define variables
%GR = 0.378;
fbracket = 39;
rbracket = 17;
GR = fbracket/rbracket;

rw=0.3; %radius wheel
lca = 0.17; %length crank arm
m = 74;
t = (time/1000)';

c1 = 0.1; %Drag for initial guess
%vcalc = (0.3344 * 2*pi*GR)./(1./(rpm/60));

% % cut data to (100:end-100)
% rpmcut = rpm(100:end-300);
% powercut = power(100:end-300);
% tcut = t(100:end-300);


% cut data to (100:end-100)
rpmcut = rpm(100:1100);
powercut = power(100:1100);
tcut = t(100:1100);

% calculated speed from gear ratio and rpm
vcalc = (rw * 2*pi*GR.*rpmcut)/60;

%% filter data
fs = 3;
[b,a] = butter(4, .1*2/fs);
v = filtfilt(b,a,vcalc);
P = filtfilt(b,a,powercut);

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
c1new = result;

%% look at accuracy of models
% %look at whole data, independent from data used for model
% rpmcut2 = rpm(100:end-100);
% powercut2 = power(100:end-100);
% tcut2 = t(100:end-100);
% vcalc2 = (rw * 2*pi*GR.*rpmcut2)/60;

rpmcut2 = rpm(100:1100);
powercut2 = power(100:1100);
tcut2 = t(100:1100);
vcalc2 = (rw * 2*pi*GR.*rpmcut2)/60;


c1new=0:0.02:0.5 ; 


%% run loop to plot Pnew with different c and save r2 and MSE1
for i=1:size(c1new,2)
    % fill in calculated variable
    Pnew = (rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*c1new(i)*v.^3;
    
    % variance
    [rsquaredreg(i),msereg(i), rmsereg(i)] = cost(P,Pnew);
end

%% plots
figure(1);
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('R2 of Different Drag Coefficients')
xlabel('Drag Coefficient') 
ylabel('R2') 
plot(c1new, rsquaredreg, 'r', 'LineWidth', 3);
hold off;

figure(2);
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('MSE of Different Drag Coefficients')
xlabel('Drag Coefficient') 
ylabel('MSE') 
hold on;
plot(c1new, msereg, 'r', 'LineWidth', 3);

