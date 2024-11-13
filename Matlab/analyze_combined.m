%{
analyze_pilot, Patrick Mayerhofer, Nov, 2021
takes all data from the subjects and puts into an excel sheet
%}
clear all; close all;

%% 
%ids = {'1_1'; '1_3'; '8_1'; '8_2'; '11_1'; '11_2'; '11_3'; '12_1'; '12_2'};
s_id = [7];
save_flag = 0;
plot_flag = 1;
load_and_save = '_onlydrag'; %_allterms, _onlydrag

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';
% load summary wind file
%dir_wind_file_summary = [dir_root, 'WindData/Summary.xlsx'];
%wind_file_summary = readtable(dir_wind_file_summary);

%% loop over each sub_trial
for subject_id  = s_id
    % load here because it will be overwritten otherwise
    dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';
    
    %% load individual wind file
    %dir_wind_file_individual = [dir_root, 'WindData/Summary_sub', num2str(subject_id), '.xlsx'];
    %wind_file_individual = readtable(dir_wind_file_individual);
    
    dir_model = [dir_root, 'ResultMatFiles/Combined/'];    
    load([dir_model, 'Subject_', mat2str(subject_id), load_and_save, '.mat']);
    r2_norm_train_mean_all(subject_id) = mean(r2_norm_train);
    r2_norm_train_std_all(subject_id) = std(r2_norm_train);
    r2_norm_test_mean_all(subject_id) = mean(r2_norm_test);
    r2_norm_test_std_all(subject_id) = std(r2_norm_test);
    
    r2_train_mean_all(subject_id) = mean(r2_train);
    r2_train_std_all(subject_id) = std(r2_train);
    r2_test_mean_all(subject_id) = mean(r2_test);
    r2_test_std_all(subject_id) = std(r2_test);
    
    r2_alternative_train_mean_all(subject_id) = mean(r2_alternative_train);
    r2_alternative_train_std_all(subject_id) = std(r2_alternative_train);
    r2_alternative_test_mean_all(subject_id) = mean(r2_alternative_test);
    r2_alternative_test_std_all(subject_id) = std(r2_alternative_test);
    
    rmse_norm_train_mean_all(subject_id) = mean(rmse_norm_train);
    rmse_norm_train_std_all(subject_id) = std(rmse_norm_train);
    rmse_norm_test_mean_all(subject_id) = mean(rmse_norm_test);
    rmse_norm_test_std_all(subject_id) = std(rmse_norm_test);
    
    rmse_train_mean_all(subject_id) = mean(rmsereg_train);
    rmse_train_std_all(subject_id) = std(rmsereg_train);
    rmse_test_mean_all(subject_id) = mean(rmsereg_test);
    rmse_test_std_all(subject_id) = std(rmsereg_test);
    
    nrmse_train_mean_all(subject_id) = mean(nrmse_train);
    nrmse_train_std_all(subject_id) = std(nrmse_train);
    nrmse_test_mean_all(subject_id) = mean(nrmse_test);
    nrmse_test_std_all(subject_id) = std(nrmse_test);

    norm_mean_err_train_mean_all(subject_id) = mean(norm_mean_err_train);
    norm_mean_err_train_std_all(subject_id) = std(norm_mean_err_train);
    norm_mean_err_test_mean_all(subject_id) = mean(norm_mean_err_test);
    norm_mean_err_test_std_all(subject_id) = std(norm_mean_err_test);
    
    %wind_mean(subject_id) = mean(wind_file_individual.mean(1:2));
    %wind_std(subject_id) = mean(wind_file_individual.std(1:2));
    
    Pnewreg_test_all{subject_id} = power_measured_norm_test;
    Pnewreg_train_all{subject_id} = power_measured_norm_train;
    data_test_all{subject_id} = data_test;
    data_train_all{subject_id} = data_train;
    
    if plot_flag
        for i = 1:length(data_test)
            fig_train = figure();
            ax = gca;
            ax.LabelFontSizeMultiplier = 2;
            ax.TitleFontSizeMultiplier = 2;
            title('Predicted Power vs Actual Power Physics-based drag and offset')
            xlabel('Time (s)') 
            ylabel('Power (W)') 
            hold on;
            plot(Pnewreg_test{i}, 'b', 'LineWidth', 3);
            plot(data_test{i}.power_filtered, 'r', 'LineWidth', 3);
            %yyaxis right 
            %plot(GR_train, 'k', 'LineWidth', 3)
            legend({'Predicted','Measured'}, 'FontSize', 16);
        end
    end
    
    
    
end


if save_flag
    data = table(nrmse_test_mean_all', nrmse_test_std_all', nrmse_train_mean_all', nrmse_train_std_all', r2_test_mean_all', r2_test_std_all', r2_train_mean_all', r2_train_std_all', r2_alternative_test_mean_all', r2_alternative_test_std_all', r2_alternative_train_mean_all', r2_alternative_train_std_all', rmse_test_mean_all', rmse_test_std_all', rmse_train_mean_all', rmse_train_std_all', norm_mean_err_test_mean_all', norm_mean_err_test_std_all', norm_mean_err_train_mean_all', norm_mean_err_train_std_all',...
        'VariableNames', {'nrmse_test_mean', 'nrmse_test_std', 'nrmse_train_mean', 'nrmse_train_std', 'r2_test_mean', 'r2_test_std', 'r2_train_mean', 'r2_train_std', 'r2_alternative_test_mean', 'r2_alternative_test_std', 'r2_alternative_train_mean', 'r2_alternative_train_std', 'rmse_test_mean', 'rmse_test_std', 'rmse_train_mean', 'rmse_train_std', 'norm_mean_err_test_mean', 'norm_mean_err_test_std', 'norm_mean_err_train_mean', 'norm_mean_err_train_std'});

    % save
    dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/MatVsPython/Combined/';
    writetable(data, [dir_root, 'Results_Summary_Matlab', load_and_save, '.csv']);
end