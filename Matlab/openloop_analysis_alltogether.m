% openloop_analysis_alltogether
% September, 2021, Patrick Mayerhofer
% one model for all subjects and trials
clear; close;
ids = {'1_1';'8_1'; '8_2'; '11_1'; '11_2'; '12_1'; '12_2'};
s_id = [1,8,11,12];
normalize_flag = 1;
save_flag = 1;
plot_flag = 1;
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

data = [];

%% prepare all my data
for subject_id = s_id
    if subject_id ==1
        m = summary_file.Weight(subject_id);
        x0 = 0.9; %Drag for initial guess
        GR1 = fbracket/summary_file.rbracket_1(subject_id);
        filename1 = ['Subject', num2str(subject_id), '_1'];
        dir_load_file1 = [dir_root, 'OpenLoop/CleanedCSV/', filename1, '.csv'];
        
        % add GR to dataset
        data1 = readtable(dir_load_file1);

        % add GR to dataset
        data1.GR = zeros(size(data1,1),1) + GR1;

        % add subject id to dataset
        data1.id = zeros(size(data1,1),1) + subject_id;

        % add sub_trial data to set with all data
        data = [data; data1];
        
    else
        %% get mass and gear ratio for sub_trial
        m = summary_file.Weight(subject_id);
        x0 = 0.9; %Drag for initial guess
        GR1 = fbracket/summary_file.rbracket_1(subject_id);
        GR2 = fbracket/summary_file.rbracket_2(subject_id);

        %% load cycling data
        filename1 = ['Subject', num2str(subject_id), '_1'];
        filename2 = ['Subject', num2str(subject_id), '_2'];
        dir_load_file1 = [dir_root, 'OpenLoop/CleanedCSV/', filename1, '.csv'];
        dir_load_file2 = [dir_root, 'OpenLoop/CleanedCSV/', filename2, '.csv'];

        % add GR to dataset
        data1 = readtable(dir_load_file1);
        data2 = readtable(dir_load_file2);

        % add GR to dataset
        data1.GR = zeros(size(data1,1),1) + GR1;
        data2.GR = zeros(size(data2,1),1) + GR2;

        % add subject id to dataset
        data1.id = zeros(size(data1,1),1) + subject_id;
        data2.id = zeros(size(data2,1),1) + subject_id;

        % add sub_trial data to set with all data
        data = [data; data1; data2];
    end
    
end

%% do calculations, leave each subject_trial out once
for i = 1:length(s_id)
    %% prepare train vs test set
    data_test = data(data.id == s_id(i), :);
    data_train = data(data.id ~= s_id(i), :);
    
    v_train = data_train.vcalc_filtered; 
    vdot_train = data_train.vcalcdot_filtered;
    P_train = data_train.power_filtered;
    time_train = data_train.time;
    rpm_train = data_train.rpm_filtered;
    GR_train = data_train.GR;
    v_vdot_GR_train = [v_train, vdot_train, GR_train];


    v_test = data_test.vcalc_filtered;
    vdot_test = data_test.vcalcdot_filtered;
    P_test = data_test.power_filtered;
    time_test = data_test.time;
    rpm_test = data_test.rpm_filtered;
    GR_test = data_test.GR;
    v_vdot_GR_test = [v_test,vdot_test, GR_test];
        
    %% regression start 
    modelfun = @(u,v_GR_train)(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3;
    u = [c1]; %unknown

    %% regression calculations
    opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
    opts = statset('nlinfit');
    opts.RobustWgtFun = 'bisquare';
    opts.MaxIter = 1000;
    [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_GR_train, P_train,modelfun,u, opts);
    c1new = result(1);

    %% look at accuracy of models
    Pnewreg_train{i} = (rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3;
    Pnewreg_test{i} = (rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3; 

    %% variance
    [r2_train(i),msereg_train(i), rmsereg_train(i)] = cost(P_train,Pnewreg_train{i});
    [r2_test(i),msereg_test(i), rmsereg_test(i)] = cost(P_test,Pnewreg_test{i});
    % r2_train = calculateR2(P_train, Pnewreg_train);

    if normalize_flag
        power_measured_norm_test{i} = (data_test.power_filtered-min(data_test.power_filtered))/(max(data_test.power_filtered)-min(data_test.power_filtered));
        power_predicted_norm_test{i} = (Pnewreg_test{i}-min(Pnewreg_test{i}))/(max(Pnewreg_test{i})-min(Pnewreg_test{i}));         
        power_measured_norm_train{i} = (data_train.power_filtered-min(data_train.power_filtered))/(max(data_train.power_filtered)-min(data_train.power_filtered));
        power_predicted_norm_train{i} = (Pnewreg_train{i}-min(Pnewreg_train{i}))/(max(Pnewreg_train{i})-min(Pnewreg_train{i}));         

        [r2_norm_test(i),msereg_norm_test(i), rmse_norm_test(i)] = cost(power_measured_norm_test{i},power_predicted_norm_test{i});
        [r2_norm_train(i),msereg_norm_train(i), rmse_norm_train(i)] = cost(power_measured_norm_train{i},power_predicted_norm_train{i});
    end
        
    %% plot 
    if plot_flag
        fig_test = figure();
        ax = gca;
        ax.LabelFontSizeMultiplier = 2;
        ax.TitleFontSizeMultiplier = 2;
        title('Predicted Power vs Actual Power normalized-test')
        xlabel('Trialtime [s]') 
        ylabel('Power [W]') 
        hold on;
        plot(power_measured_norm_test{i}, 'b', 'LineWidth', 3);
        plot(power_predicted_norm_test{i}, 'r', 'LineWidth', 3);
        yyaxis right 
        plot(GR_test, 'k', 'LineWidth', 3)
        legend({'Measured','Predicted'}, 'FontSize', 16);

        fig_train = figure();
        ax = gca;
        ax.LabelFontSizeMultiplier = 2;
        ax.TitleFontSizeMultiplier = 2;
        title('Predicted Power vs Actual Power normalized-train')
        xlabel('Trialtime [s]') 
        ylabel('Power [W]') 
        hold on;
        plot(power_measured_norm_train{i}, 'b', 'LineWidth', 3);
        plot(power_predicted_norm_train{i}, 'r', 'LineWidth', 3);
        yyaxis right 
        plot(GR_train, 'k', 'LineWidth', 3)
        legend({'Measured','Predicted'}, 'FontSize', 16);
    end
 end
     % if in a loop, do this only in the very end. - still needs to be changed
    % to wherever we want to save it 
if save_flag
    dir_save = [dir_root, 'OpenLoop/ResultMatFiles/Pilot/Combined/Subject_', num2str(subject_id), '.mat'];
    save(dir_save, 'data_train', 'data_test', 'Pnewreg_train', 'Pnewreg_test', 'power_measured_norm_train', 'power_measured_norm_test', 'power_predicted_norm_train', 'power_predicted_norm_test', 'r2_norm_train', 'r2_norm_test', 'r2_train', 'r2_test', 'rmse_norm_train', 'rmse_norm_test', 'rmsereg_train', 'rmsereg_test');
    %writetable(local_summary_file, dir_summary_file);

end