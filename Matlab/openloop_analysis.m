% openloop_analysis
% September, 2021, Patrick Mayerhofer

clear all;
close all;
%% Changeable variables
s_id = [2];
normalize_flag = 1;
plot_flag = 1;
plot_flag_2 = 1;
save_flag = 0;
c1 = 1;



%% load summary file
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/';
% excel summary file 
dir_summary_file = [dir_root, 'Summary.xlsx'];
summary_file = readtable(dir_summary_file);
local_summary_file = summary_file; % because the mat file also has summary file saved, and overwrites it otherwise when we load it

%% bikespecs
rw = 0.3; %works best with actual gpsspeed
%rw=0.325; %radius wheel original
lca = 0.17; %length crank arm
fbracket = 39;

for subject_id = s_id
    m = summary_file.Weight(subject_id);

    for trial_id = 1:3
        x0 = 0.9; %Drag for initial guess

        %% rbracket and gear ratio
        if trial_id == 1
            rbracket = summary_file.rbracket_1(subject_id);
        elseif trial_id == 2
            rbracket = summary_file.rbracket_2(subject_id);
        elseif trial_id == 3
            rbracket = summary_file.rbracket_3(subject_id);
        end

        GR = fbracket/rbracket;


        %% load csv files
        filename = ['Subject', num2str(subject_id), '_', num2str(trial_id)];
        dir_load_file_train = [dir_root, 'OpenLoop/CleanedCSV/', filename, '_train.csv'];
        dir_load_file_test = [dir_root, 'OpenLoop/CleanedCSV/', filename, '_test.csv'];
        data_train = readtable(dir_load_file_train);
        data_test = readtable(dir_load_file_test);

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
        if plot_flag
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
        end

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

        %% variance
        [r2_train,msereg_train, rmsereg_train] = cost(P_train,Pnewreg_train);
        [r2_test,msereg_test, rmsereg_test] = cost(P_test,Pnewreg_test);
        % r2_train = calculateR2(P_train, Pnewreg_train);

        if normalize_flag
            power_measured_norm_test = (data_test.power_filtered-min(data_test.power_filtered))/(max(data_test.power_filtered)-min(data_test.power_filtered));
            power_predicted_norm_test = (Pnewreg_test-min(Pnewreg_test))/(max(Pnewreg_test)-min(Pnewreg_test));         
            power_measured_norm_train = (data_train.power_filtered-min(data_train.power_filtered))/(max(data_train.power_filtered)-min(data_train.power_filtered));
            power_predicted_norm_train = (Pnewreg_train-min(Pnewreg_train))/(max(Pnewreg_train)-min(Pnewreg_train));         

            [r2_norm_test,msereg_norm_test, rmse_norm_test] = cost(power_measured_norm_test,power_predicted_norm_test);
            [r2_norm_train,msereg_norm_train, rmse_norm_train] = cost(power_measured_norm_train,power_predicted_norm_train);


            if plot_flag_2
                fig_test = figure();
                ax = gca;
                ax.LabelFontSizeMultiplier = 2;
                ax.TitleFontSizeMultiplier = 2;
                title('Predicted Power vs Actual Power normalized-test')
                xlabel('Trialtime [s]') 
                yyaxis left;
                ylabel('Power [W]') 
                hold on;
                plot(power_measured_norm_test, 'b', 'LineWidth', 3);
                plot(power_predicted_norm_test, 'r', 'LineWidth', 3);
                legend({'Measured','Predicted'}, 'FontSize', 16);

                fig_train = figure();
                ax = gca;
                ax.LabelFontSizeMultiplier = 2;
                ax.TitleFontSizeMultiplier = 2;
                title('Predicted Power vs Actual Power normalized-train')
                xlabel('Trialtime [s]') 
                ylabel('Power [W]') 
                hold on;
                plot(power_measured_norm_train, 'b', 'LineWidth', 3);
                plot(power_predicted_norm_train, 'r', 'LineWidth', 3);
                legend({'Measured','Predicted'}, 'FontSize', 16);

                if save_flag
                    filename_test = [dir_root, 'Graphs/Subject', num2str(subject_id), '_', num2str(trial_id), '_test.fig'];
                    saveas(fig_test, filename_test);
                    filename_train = [dir_root, 'Graphs/Subject', num2str(subject_id), '_', num2str(trial_id), '_train.fig'];
                    saveas(fig_train, filename_train);
                end
            end
        end

        %% fill summary file and if true, save data to excel sheet

        local_summary_file = save_data(local_summary_file, subject_id, trial_id, r2_norm_train,r2_norm_test,rmse_norm_train,rmse_norm_test);
    end
end

% if in a loop, do this only in the very end.
if save_flag
    
    writetable(local_summary_file, dir_summary_file);
    
end
