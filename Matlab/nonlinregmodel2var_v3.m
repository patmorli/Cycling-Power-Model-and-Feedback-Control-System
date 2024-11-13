%% nonlinear regression model, find two unknown variables 
%% normalized data
%% May, 2019
clear all;
close all

%% define variables and flags
filename = 'Oct19OL_Patv2';


rw=0.33; %radius wheel
lca = 0.17; %length crank arm
m = 74;
c1 =1;
var = 1;
x0 = 0.9; %Drag for initial guess
flag_plot = 1;

opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;

%% directions
dir_root = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Cycling Project/2023 Model and Feedback/ModelOptimization/';
dir_data = [dir_root, 'Data/']; 



%% load a file
load([dir_data, filename]);


%cut data to (100:end-100)
t = (time/1000)';
rpmcut = rpm(85:2830);
powercut = power(85:2830);
tcut = t(85:2830);



% calculated speed from gear ratio and rpm
vcalc = (rw * 2*pi*GR.*rpmcut)/60;

%% filter data
fs = 3;
[b,a] = butter(4, .1*2/fs);
v = filtfilt(b,a,vcalc);
P = filtfilt(b,a,powercut);


%% plot data of optimization
if flag_plot
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
end

%% derive
dv = nanmean([diff([v NaN]); diff([NaN v])]); % central difference
dt = nanmean([diff([tcut NaN]); diff([NaN tcut])]);
vdot = dv./dt;



%% regression start 
u = [c1]; %unknown
% old model
modelfun_old = @(u,v)(rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*u(1)*v.^3;
[result_old,R_old,J_old,CovB_old,MSE_old,ErrorModelInfo_old] = nlinfit(v,P,modelfun_old,u,opts);
c1new_old = result_old(1);



% new model
v_vdot = [v; vdot];
modelfun_new = @(u,v_train)m.*v_vdot(1,:).*v_vdot(2,:) + u(1).*v_vdot(1,:).^3;
[result_new,R_new,J_new,CovB_new,MSE_new,ErrorModelInfo_new] = nlinfit(v_vdot, P,modelfun_new,u,opts);
c1new_new = result_new(1);


%% fill in calculated variable
% old model
Pnewreg_old = (rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*c1new_old*v.^3; 

% new model
% Pnewreg_new_inertialpower = m.*v_vdot1(1,:).*v_vdot1(2,:);
% Pnewreg_new_dragpower = c1new_new1.*v_vdot1(1,:).^3;
% Pnewreg_new = m.*v_vdot1(1,:).*v_vdot1(2,:) + c1new_new1.*v_vdot1(1,:).^3;

Pnewreg_new = m.*v_vdot(1,:).*v_vdot(2,:) + c1new_new.*v_vdot(1,:).^3;
 
 
 
%% variance
[rsquaredreg_old,msereg_old, rmsereg_old, mae_old] = cost(P,Pnewreg_old);
[rsquaredreg_new,msereg_new, rmsereg_new, mae_new] = cost(P,Pnewreg_new);
nRMSE_new = rmsereg_new/(mean(P));
nRMSE_old = rmsereg_old/(mean(P));
nmae_new = mae_new/(mean(P));
nmae_old = mae_old/(mean(P));

[rsquaredreg_between_models,msereg_between_models, rmsereg_between_models] = cost(Pnewreg_old,Pnewreg_new);
nRMSE_between_models = rmsereg_between_models/mean(P);

%% noise 
data_std = std(rpmcut(2120:2280));


if flag_plot
    figure();
    ax = gca;
    ax.LabelFontSizeMultiplier = 2;
    ax.TitleFontSizeMultiplier = 2;
    title('Power of models and measured')
    xlabel('Trial time [s]') 
    ylabel('Power [W]') 
    hold on;
    plot(tcut, Pnewreg_old, 'r', 'LineWidth', 3);
    plot(tcut, Pnewreg_new, 'b', 'LineWidth', 3);
    plot(tcut,P, 'g', 'LineWidth', 3)
    legend({'old model power','new model power', 'measured power'}, 'FontSize', 16);
end
