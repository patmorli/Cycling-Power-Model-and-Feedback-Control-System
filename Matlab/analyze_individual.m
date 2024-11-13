%{
analyze_pilot, Patrick Mayerhofer, Nov, 2021
takes all data from the 4 pilot subjects and puts into an excel sheet
%}
clear all; close all;

save_flag = 1; %might need change
per_subject_save_flag = 0;
plot_flag = 0;
load_and_save = '_onlydrag'; %_allterms, _onlydrag


%% 
%ids = {'1_1'; '1_3'; '8_1'; '8_2'; '11_1'; '11_2'; '11_3'; '12_1'; '12_2'};
% can just change this to two for loops with subjects and trials, because
% when it is not pilot, all subjects have trial 1-3
% ids = {'3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3', '6_1', '6_2', '6_3', ... 
%     '7_1', '7_2', '7_3', '9_1', '9_2', '9_3', '10_1', '10_2', '10_3', ...
%      '13_1', '13_2', '13_3',};
% ids = {'9_1', '9_2', '11_1', '11_2', '12_1', '12_2'};
% ids = {'9_1'};

s_id = [2,3,4,5,6,7,8,9,10,11,12,13];
t_id = [1,2];

 
 
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';
% load summary wind file
dir_wind_file_summary = [dir_root, 'WindData/Summary.xlsx'];
wind_file_summary = readtable(dir_wind_file_summary);
p = 0;
%% loop over each sub_trial
for subject_id = s_id
    for trial_id = t_id
        p = p + 1;
        % load here because it will be overwritten otherwise
        dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';

        %% load individual wind file
        %dir_wind_file_individual = [dir_root, 'WindData/Summary_sub', ids{s}(1:end-2), '.xlsx'];
        %wind_file_individual = readtable(dir_wind_file_individual);

        dir_model = [dir_root, 'ResultMatFiles/Individual/'];    
        load([dir_model, 'Subject', mat2str(subject_id), '_', mat2str(trial_id), load_and_save, '.mat']);
        trial_r2_norm_train_mean_all(p) = mean(r2_norm_train);
        trial_r2_norm_train_std_all(p) = std(r2_norm_train);
        trial_r2_norm_test_mean_all(p) = mean(r2_norm_test);
        trial_r2_norm_test_std_all(p) = std(r2_norm_test);

        trial_r2_train_mean_all(p) = mean(r2_train);
        trial_r2_train_std_all(p) = std(r2_train);
        trial_r2_test_mean_all(p) = mean(r2_test);
        trial_r2_test_std_all(p) = std(r2_test);
        
        trial_r2_alternative_train_mean_all(p) = mean(r2_alternative_train);
        trial_r2_alternative_train_std_all(p) = std(r2_alternative_train);
        trial_r2_alternative_test_mean_all(p) = mean(r2_alternative_test);
        trial_r2_alternative_test_std_all(p) = std(r2_alternative_test);

        trial_rmse_norm_train_mean_all(p) = mean(rmse_norm_train);
        trial_rmse_norm_train_std_all(p) = std(rmse_norm_train);
        trial_rmse_norm_test_mean_all(p) = mean(rmse_norm_test);
        trial_rmse_norm_test_std_all(p) = std(rmse_norm_test);

        trial_rmse_train_mean_all(p) = mean(rmsereg_train);
        trial_rmse_train_std_all(p) = std(rmsereg_train);
        trial_rmse_test_mean_all(p) = mean(rmsereg_test);
        trial_rmse_test_std_all(p) = std(rmsereg_test);

        trial_nrmse_train_mean_all(p) = mean(nrmse_train);
        trial_nrmse_train_std_all(p) = std(nrmse_train);
        trial_nrmse_test_mean_all(p) = mean(nrmse_test);
        trial_nrmse_test_std_all(p) = std(nrmse_test);

        trial_norm_mean_err_train_mean_all(p) = mean(norm_mean_err_train);
        trial_norm_mean_err_train_std_all(p) = std(norm_mean_err_train);
        trial_norm_mean_err_test_mean_all(p) = mean(norm_mean_err_test);
        trial_norm_mean_err_test_std_all(p) = std(norm_mean_err_test);

        %wind_mean(subject_id) = mean(wind_file_individual.mean(1:2));
        %wind_std(subject_id) = mean(wind_file_individual.std(1:2));

        Pnewreg_test_all{p} = power_measured_norm_test;
        Pnewreg_train_all{p} = power_measured_norm_train;
        data_test_all{p} = data_test;
        data_train_all{p} = data_train;
        
        id_names{p} = [mat2str(subject_id), '_', mat2str(trial_id)];

        if plot_flag
            for i = 1:length(data_test)
                fig_train = figure();
                ax = gca;
                ax.LabelFontSizeMultiplier = 2;
                ax.TitleFontSizeMultiplier = 2;
                title(['Predicted Power vs Actual Power Physics-based drag and offset_' id_names(p)])
                xlabel('Trialtime []') 
                ylabel('Power [W]') 
                hold on;
                plot(Pnewreg_test{i}, 'b', 'LineWidth', 3);
                plot(data_test{i}.power_filtered, 'r', 'LineWidth', 3);
                %yyaxis right 
                %plot(GR_train, 'k', 'LineWidth', 3)
                legend({'Predicted','Measured'}, 'FontSize', 16);
            end
        end
    end
    
    subject_r2_train_mean_all(p/2) = mean(trial_r2_train_mean_all(end-1:end));
    subject_r2_train_std_all(p/2) = mean(trial_r2_train_std_all(end-1:end));
    subject_r2_test_mean_all(p/2) = mean(trial_r2_test_mean_all(end-1:end));
    subject_r2_test_std_all(p/2) = mean(trial_r2_test_std_all(end-1:end));

    subject_r2_alternative_train_mean_all(p/2) = mean(trial_r2_alternative_train_mean_all(end-1:end));
    subject_r2_alternative_train_std_all(p/2) = mean(trial_r2_alternative_train_std_all(end-1:end));
    subject_r2_alternative_test_mean_all(p/2) = mean(trial_r2_alternative_test_mean_all(end-1:end));
    subject_r2_alternative_test_std_all(p/2) = mean(trial_r2_alternative_test_std_all(end-1:end));
    
    subject_rmse_train_mean_all(p/2) = mean(trial_rmse_train_mean_all(end-1:end));
    subject_rmse_train_std_all(p/2) = mean(trial_rmse_train_std_all(end-1:end));
    subject_rmse_test_mean_all(p/2) = mean(trial_rmse_test_mean_all(end-1:end));
    subject_rmse_test_std_all(p/2) = mean(trial_rmse_test_std_all(end-1:end));

    subject_nrmse_train_mean_all(p/2) = mean(trial_nrmse_train_mean_all(end-1:end));
    subject_nrmse_train_std_all(p/2) = mean(trial_nrmse_train_std_all(end-1:end));
    subject_nrmse_test_mean_all(p/2) = mean(trial_nrmse_test_mean_all(end-1:end));
    subject_nrmse_test_std_all(p/2) = mean(trial_nrmse_test_std_all(end-1:end));

    subject_norm_mean_err_train_mean_all(p/2) = mean(trial_norm_mean_err_train_mean_all(end-1:end));
    subject_norm_mean_err_train_std_all(p/2) = mean(trial_norm_mean_err_train_std_all(end-1:end));
    subject_norm_mean_err_test_mean_all(p/2) = mean(trial_norm_mean_err_test_mean_all(end-1:end));
    subject_norm_mean_err_test_std_all(p/2) = mean(trial_norm_mean_err_test_std_all(end-1:end));
    
end

data = table(id_names', trial_nrmse_test_mean_all', trial_nrmse_test_std_all', trial_nrmse_train_mean_all', trial_nrmse_train_std_all', trial_r2_test_mean_all', trial_r2_test_std_all', trial_r2_train_mean_all', trial_r2_train_std_all', trial_r2_alternative_test_mean_all', trial_r2_alternative_test_std_all', trial_r2_alternative_train_mean_all', trial_r2_alternative_train_std_all', trial_rmse_test_mean_all', trial_rmse_test_std_all', trial_rmse_train_mean_all', trial_rmse_train_std_all',trial_norm_mean_err_test_mean_all', trial_norm_mean_err_test_std_all', trial_norm_mean_err_train_mean_all', trial_norm_mean_err_train_std_all', ...
     'VariableNames', {'ID','nrmse_test_mean', 'nrmse_test_std','nrmse_train_mean_all', 'nrmse_train_std_all', 'r2_test_mean', 'r2_test_std', 'r2_train_mean', 'r2_train_std', 'r2_alternative_test_mean', 'r2_alternative_test_std', 'r2_alternative_train_mean', 'r2_alternative_train_std', 'rmse_test_mean', 'rmse_test_std', 'rmse_train_mean', 'rmse_train_std', 'norm_mean_err_test_mean', 'norm_mean_err_test_std', 'norm_mean_err_train_mean', 'norm_mean_err_train_std'});

data_per_subject = table(s_id', subject_nrmse_test_mean_all', subject_nrmse_test_std_all', subject_nrmse_train_mean_all', subject_nrmse_train_std_all', subject_r2_test_mean_all', subject_r2_test_std_all', subject_r2_train_mean_all', subject_r2_train_std_all', subject_r2_alternative_test_mean_all', subject_r2_alternative_test_std_all', subject_r2_alternative_train_mean_all', subject_r2_alternative_train_std_all', subject_rmse_test_mean_all', subject_rmse_test_std_all', subject_rmse_train_mean_all', subject_rmse_train_std_all',subject_norm_mean_err_test_mean_all', subject_norm_mean_err_test_std_all', subject_norm_mean_err_train_mean_all', subject_norm_mean_err_train_std_all',...
     'VariableNames', {'ID','nrmse_test_mean', 'nrmse_test_std','nrmse_train_mean_all', 'nrmse_train_std_all', 'r2_test_mean', 'r2_test_std', 'r2_train_mean', 'r2_train_std', 'r2_alternative_test_mean', 'r2_alternative_test_std', 'r2_alternative_train_mean', 'r2_alternative_train_std', 'rmse_test_mean', 'rmse_test_std', 'rmse_train_mean', 'rmse_train_std', 'norm_mean_err_test_mean', 'norm_mean_err_test_std', 'norm_mean_err_train_mean', 'norm_mean_err_train_std'});
 
if save_flag 
    % save
    dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/MatVsPython/Individual/';
    writetable(data, [dir_root, 'Results_Summary_Matlab', load_and_save, '.csv']);
end

if per_subject_save_flag
    dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/MatVsPython/Individual/';
    writetable(data_per_subject, [dir_root, 'per_subject_Results_Summary_Matlab', load_and_save, '.csv']);
    
end

% if plot_flag
%     for i = 1:3
%         fig_test = figure();
%         ax = gca;
%         ax.LabelFontSizeMultiplier = 2;
%         ax.TitleFontSizeMultiplier = 2;
%         title('Predicted Power vs Actual Power normalized-test')
%         xlabel('Trialtime [s]') 
%         ylabel('Power [W]') 
%         hold on;
%         plot(power_measured_norm_test{i}, 'b', 'LineWidth', 3);
%         plot(power_predicted_norm_test{i}, 'r', 'LineWidth', 3);
%         legend({'Measured','Predicted'}, 'FontSize', 16);
%         
%         if 0
%             fig_train = figure();
%             ax = gca;
%             ax.LabelFontSizeMultiplier = 2;
%             ax.TitleFontSizeMultiplier = 2;
%             title('Predicted Power vs Actual Power normalized-train')
%             xlabel('Trialtime [s]') 
%             ylabel('Power [W]') 
%             hold on;
%             plot(power_measured_norm_train{i}, 'b', 'LineWidth', 3);
%             plot(power_predicted_norm_train{i}, 'r', 'LineWidth', 3);
%             legend({'Measured','Predicted'}, 'FontSize', 16);
%         end
%     end
%end
       
