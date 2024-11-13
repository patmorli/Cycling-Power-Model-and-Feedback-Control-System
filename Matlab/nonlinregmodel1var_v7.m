% nonlinear regression model, find one unknown variable
% with data of whole trial
% Nov 13, 2018

clear all;
close all


%% load a file
filename = 'Subject4_2';
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';
dir_load_file_train = [dir_root, 'CleanedCSV/', filename, '_train.csv'];
dir_load_file_test = [dir_root, 'CleanedCSV/', filename, '_test.csv'];
data_train = readtable(dir_load_file_train);
data_test = readtable(dir_load_file_test);
load([dir_root, 'MatFiles/', filename])

%% define variables
normalize_flag = 1;
c1 =1;

x0 = 0.9; %Drag for initial guess
%vcalc = (0.3344 * 2*pi*GR)./(1./(rpm/60));

%% get variables for calculation
%v_train = data_train.gpsSpeed_filtered;
%vdot_train = data_train.vdot_filtered;
v_train = data_train.vcalc_filtered;
vdot_train = data_train.vcalcdot_filtered;
P_train = data_train.power_filtered;
time_train = data_train.time;
rpm_train = data_train.rpm_filtered;




v_test = data_test.vcalc_filtered;
vdot_test = data_test.vcalcdot_filtered;
P_test = data_test.power_filtered;
time_test = data_test.time;
rpm_test = data_test.rpm_filtered;



%% plot data of optimization
figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power and RPM of data used for optimization')
xlabel('Trialtime [s]') 

hold on;
yyaxis left; plot(time_train, rpm_train, 'b', 'LineWidth', 3);
ylabel('Cadence [RPM]');
yyaxis right; plot(time_train, P_train, 'R', 'LineWidth', 3);
ylabel('Power [W]');
legend({'Cadence','Power'}, 'FontSize', 16);




%% regression start 
modelfun = @(u,v_train)(rw/lca)*GR*m*vdot_train.*v_train + (rw/lca)*GR*u(1)*v_train.^3;
u = [c1]; %unknown

%% regression calculations
opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
 [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_train,P_train,modelfun,u, opts);
 c1new = result(1);


 
 %% look at accuracy of models
 Pnewreg_train = (rw/lca)*GR*m*vdot_train.*v_train + (rw/lca)*GR*c1new*v_train.^3; 
 Pnewreg_test = (rw/lca)*GR*m*vdot_test.*v_test + (rw/lca)*GR*c1new*v_test.^3; 

 %% normalize for comparison
 
 
 
 %% variance
[r2_train,msereg_train, rmsereg_train] = cost(P_train,Pnewreg_train);
[r2_test,msereg_test, rmsereg_test] = cost(P_test,Pnewreg_test);

if normalize_flag
    power_measured_norm = (data_test.power_filtered-min(data_test.power_filtered))/(max(data_test.power_filtered)-min(data_test.power_filtered));
    power_predicted_norm = (Pnewreg_test-min(Pnewreg_test))/(max(Pnewreg_test)-min(Pnewreg_test));         
    
    [r2_norm_test,msereg_norm_test, rmsereg_norm_test] = cost(power_measured_norm,power_predicted_norm);
    
    figure(100);
    ax = gca;
    ax.LabelFontSizeMultiplier = 2;
    ax.TitleFontSizeMultiplier = 2;
    title('Comparison Derived Power and Actual Power normalized-test')
    xlabel('Trialtime [s]') 
    ylabel('Power [W]') 
    hold on;
    plot(power_measured_norm, 'b', 'LineWidth', 3);
    plot(power_predicted_norm, 'r', 'LineWidth', 3);
    legend({'Measured','Predicted'}, 'FontSize', 16);
end

if plot_flag
    figure();
    ax = gca;
    ax.LabelFontSizeMultiplier = 2;
    ax.TitleFontSizeMultiplier = 2;
    title('Comparison Derived Power and Actual Power-train')
    xlabel('Trialtime [s]') 
    ylabel('Power [W]') 
    hold on;
    plot(Pnewreg_train, 'r', 'LineWidth', 3);
    plot(P_train, 'b', 'LineWidth', 3);
    legend({'Simulated','Measured'}, 'FontSize', 16);

    figure();
    ax = gca;
    ax.LabelFontSizeMultiplier = 2;
    ax.TitleFontSizeMultiplier = 2;
    title('Comparison Derived Power and Actual Power-test')
    xlabel('Trialtime [s]') 
    ylabel('Power [W]') 
    hold on;
    plot(Pnewreg_test, 'r', 'LineWidth', 3);
    plot(P_test, 'b', 'LineWidth', 3);
    legend({'Simulated','Measured'}, 'FontSize', 16);
end

%save(filename);