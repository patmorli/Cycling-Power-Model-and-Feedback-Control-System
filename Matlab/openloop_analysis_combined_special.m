% openloop_analysis_combined
% September, 2021, Patrick Mayerhofer

clear all;
close all;
%% Changeable variables
% try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
s_id = [4];
t_id = [1,2];
normalize_flag = 1;
save_flag = 0;
plot_flag = 0;
acc_term_flag = 0;
offset_term_flag = 0;
c1 = 1;
acc_term = 1;
offset_term = 1;
opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
bike_and_equipment_weight = 13;



%% load summary file
dir_root = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Cycling Project/2021/SubjectData/';
% excel summary file 
dir_summary_file = [dir_root, 'Summary.xlsx'];
summary_file = readtable(dir_summary_file,  'Format', 'auto');
local_summary_file = summary_file; % because the mat file also has summary file saved, and overwrites it otherwise when we load it

%% bikespecs
rw = 0.3; %works best with actual gpsspeed
%rw=0.325; %radius wheel original
lca = 0.17; %length crank arm
fbracket = 39;

drag_number = [];
for subject_id = s_id
    m = summary_file.Weight(subject_id)+bike_and_equipment_weight;
    x0 = 0.9; %Drag for initial guess
    GR1 = fbracket/summary_file.rbracket_1(subject_id);
    GR2 = fbracket/summary_file.rbracket_2(subject_id);
    
    
    
    %% load cycling data
    filename1 = ['Subject', num2str(subject_id), '_1'];
    filename2 = ['Subject', num2str(subject_id), '_2'];
    dir_load_file1 = [dir_root, 'OpenLoop/CleanedCSV/', filename1, '.csv'];
    dir_load_file2 = [dir_root, 'OpenLoop/CleanedCSV/', filename2, '.csv'];
    data1 = readtable(dir_load_file1);
    data2 = readtable(dir_load_file2);
    
    % add GR to dataset
    data1.GR = zeros(size(data1,1),1) + GR1;
    data2.GR = zeros(size(data2,1),1) + GR2;

    %create indizes for 3 same-sized test sets from each trial
    l1_ = size(data1,1);
    l1_3 = round(l1_/3);
    increments1 = [0, l1_3, l1_3*2, l1_];

    l2_ = size(data2,1);
    l2_3 = round(l2_/3);
    increments2 = [0, l2_3,l2_3*2, l2_];
    increments2 = increments2 + l1_;
    increments2(1) = [];
    increments = [increments1, increments2];
   
    data = [data1; data2];
     
    %% do calculation for 6 different datasets (3 in each trial)
     for i = 1:6
        ones_matrix_test = zeros(increments(7),1);
        ones_matrix_train = ones(increments(7),1); 
        ones_matrix_test(increments(i)+1:increments(i+1)) = 1;
        ones_matrix_train(increments(i)+1:increments(i+1)) = 0;
        
        %divide in train and data set
        data_test{i} = data(ones_matrix_test>0,:);
        data_train{i} = data(ones_matrix_train>0,:);
        
        %% prepare dataset 
        v_train = data_train{i}.vcalc_filtered; 
        vdot_train = data_train{i}.vcalcdot_filtered;
        P_train = data_train{i}.power_filtered;
        time_train = data_train{i}.time;
        rpm_train = data_train{i}.rpm_filtered;
        GR_train = data_train{i}.GR;
        v_vdot_train = [v_train, vdot_train];


        v_test = data_test{i}.vcalc_filtered;
        vdot_test = data_test{i}.vcalcdot_filtered;
        P_test = data_test{i}.power_filtered;
        time_test = data_test{i}.time;
        rpm_test = data_test{i}.rpm_filtered;
        GR_test = data_test{i}.GR;
        v_vdot_test = [v_test,vdot_test];
        
        %% regression start 
        if acc_term_flag == 0 & offset_term_flag == 0 % original, only the drag term
            load_and_save = '_onlydrag';
            modelfun = @(u,v_train)m.*v_vdot_train(:,1).*v_vdot_train(:,2) + u(1).*v_vdot_train(:,1).^3;
            %modelfun = @(u,v_GR_train)(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3;
            u = [c1]; %unknown
            
            [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_train, P_train,modelfun,u, opts);
            c1new = result(1);
            
            d(i) = c1new;
            
            %% predict power with optimized variable(s)
            Pnewreg_train{i} = m.*v_vdot_train(:,1).*v_vdot_train(:,2) + c1new.*v_vdot_train(:,1).^3;
            Pnewreg_test{i} = m.*v_vdot_test(:,1).*v_vdot_test(:,2) + c1new.*v_vdot_test(:,1).^3;
            
%             Pnewreg_train{i} = (rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3;
%             Pnewreg_test{i} = (rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3; 
        end
        
        if acc_term_flag == 1 & offset_term_flag == 0 % drag and acceleration term c
            load_and_save = '_dragandaccterm';
                modelfun = @(u,v_train)u(2)*m.*v_vdot_train(:,1).*v_vdot_train(:,2) + u(1).*v_vdot_train(:,1).^3;
                %modelfun = @(u,v_GR_train)u(2)*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3;
                u = [c1 acc_term]; %unknown

                [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_train, P_train,modelfun,u, opts);
                c1new = result(1);
                acc_termnew = result(2);

                d(i) = c1new;
                a(i) = acc_termnew;

                %% predict power with optimized variable(s)
                Pnewreg_train{i} = acc_termnew*m.*v_vdot_train(:,1).*v_vdot_train(:,2) + c1new.*v_vdot_train(:,1).^3;
                Pnewreg_test{i} = acc_termnew*m.*v_vdot_test(:,1).*v_vdot_test(:,2) + c1new.*v_vdot_test(:,1).^3;
%                 Pnewreg_train{i} = acc_termnew*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3;
%                 Pnewreg_test{i} = acc_termnew*(rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3; 
        end
        
        if acc_term_flag == 1 & offset_term_flag == 1 % drag, acceleration, and offset term
            load_and_save = '_allterms';
            modelfun = @(u,v_train)u(2)*m.*v_vdot_train(:,1).*v_vdot_train(:,2) + u(1).*v_vdot_train(:,1).^3 + u(3);
            %modelfun = @(u,v_GR_train)u(2)*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3 + u(3);
            u = [c1 acc_term offset_term]; %unknown

            [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_train, P_train,modelfun,u, opts);
            c1new = result(1);
            acc_termnew = result(2);
            offset_termnew = result(3);

            d(i) = c1new;
            a(i) = acc_termnew;
            o(i) = offset_termnew;

            %% predict power with optimized variable(s)
            Pnewreg_train{i} = acc_termnew*m.*v_vdot_train(:,1).*v_vdot_train(:,2) + c1new.*v_vdot_train(:,1).^3 + offset_termnew;
            Pnewreg_test{i} = acc_termnew*m.*v_vdot_test(:,1).*v_vdot_test(:,2) + c1new.*v_vdot_test(:,1).^3 + offset_termnew;
%                 Pnewreg_train{i} = acc_termnew*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3+offset_termnew;
%                 Pnewreg_test{i} = acc_termnew*(rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3+offset_termnew; 
        end
        
        if acc_term_flag == 0 & offset_term_flag == 1 % drag and offset term
            load_and_save = '_dragandoffsetterm';
            modelfun = @(u,v_train)m.*v_vdot_train(:,1).*v_vdot_train(:,2) + u(1).*v_vdot_train(:,1).^3 + u(2);
            %modelfun = @(u,v_GR_train)(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3 + u(2);
            u = [c1 offset_term]; %unknown

            [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_train, P_train,modelfun,u, opts);
            c1new = result(1);
            offset_termnew = result(2);

            d(i) = c1new;
            o(i) = offset_termnew;

            %% predict power with optimized variable(s)
            Pnewreg_train{i} = m.*v_vdot_train(:,1).*v_vdot_train(:,2) + c1new.*v_vdot_train(:,1).^3 + offset_termnew;
            Pnewreg_test{i} = m.*v_vdot_test(:,1).*v_vdot_test(:,2) + c1new.*v_vdot_test(:,1).^3 + offset_termnew;
%                 Pnewreg_train{i} = (rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3+offset_termnew;
%                 Pnewreg_test{i} = (rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3+offset_termnew; 
        end


        %% variance
        % reg in the name not needed, but left it so that I do not have to
        % change it everywhere
        [r2_train(i), r2_alternative_train(i), msereg_train(i), rmsereg_train(i), nrmse_train(i), norm_mean_err_train(i)] = cost(P_train,Pnewreg_train{i});
        [r2_test(i),r2_alternative_test(i), msereg_test(i), rmsereg_test(i), nrmse_test(i), norm_mean_err_test(i)] = cost(P_test,Pnewreg_test{i});
        % r2_train = calculateR2(P_train, Pnewreg_train);
        
        if normalize_flag
            power_measured_norm_test{i} = (data_test{i}.power_filtered-min(data_test{i}.power_filtered))/(max(data_test{i}.power_filtered)-min(data_test{i}.power_filtered));
            power_predicted_norm_test{i} = (Pnewreg_test{i}-min(Pnewreg_test{i}))/(max(Pnewreg_test{i})-min(Pnewreg_test{i}));         
            power_measured_norm_train{i} = (data_train{i}.power_filtered-min(data_train{i}.power_filtered))/(max(data_train{i}.power_filtered)-min(data_train{i}.power_filtered));
            power_predicted_norm_train{i} = (Pnewreg_train{i}-min(Pnewreg_train{i}))/(max(Pnewreg_train{i})-min(Pnewreg_train{i}));         

            [r2_norm_test(i), r2_norm_alternative_test(i), msereg_norm_test(i), rmse_norm_test(i), norm_mean_err_norm_test(i)] = cost(power_measured_norm_test{i},power_predicted_norm_test{i});
            [r2_norm_train(i), r2_norm_alternative_train(i) ,msereg_norm_train(i), rmse_norm_train(i), norm_mean_err_norm_train(i)] = cost(power_measured_norm_train{i},power_predicted_norm_train{i});
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
            plot(Pnewreg_test{i}, 'b', 'LineWidth', 3);
            plot(P_test, 'r', 'LineWidth', 3);
            yyaxis right 
            plot(GR_test, 'k', 'LineWidth', 3)
            legend({'Predicted','Measured'}, 'FontSize', 16);

            fig_train = figure();
            ax = gca;
            ax.LabelFontSizeMultiplier = 2;
            ax.TitleFontSizeMultiplier = 2;
            title('Predicted Power vs Actual Power normalized-train')
            xlabel('Trialtime [s]') 
            ylabel('Power [W]') 
            hold on;
            plot(Pnewreg_train{i}, 'b', 'LineWidth', 3);
            plot(P_train, 'r', 'LineWidth', 3);
            yyaxis right 
            plot(GR_train, 'k', 'LineWidth', 3)
            legend({'Predicted','Measured'}, 'FontSize', 16);
        end
     end
     drag_number = [drag_number d];
     
     % if in a loop, do this only in the very end. - still needs to be changed
    % to wherever we want to save it 
    if save_flag
        dir_save = [dir_root, 'OpenLoop/ResultMatFiles/Combined/Subject_', num2str(subject_id), load_and_save, '.mat'];
        save(dir_save, 'data_train', 'data_test', 'Pnewreg_train', 'Pnewreg_test', 'power_measured_norm_train', 'power_measured_norm_test', 'power_predicted_norm_train', 'power_predicted_norm_test', 'r2_norm_train', 'r2_norm_test', 'r2_train', 'r2_alternative_train', 'r2_test', 'r2_alternative_test', 'rmse_norm_train', 'rmse_norm_test', 'rmsereg_train', 'rmsereg_test', 'nrmse_train', 'nrmse_test', 'norm_mean_err_train', 'norm_mean_err_test');
        %writetable(local_summary_file, dir_summary_file);
    
    end
end

disp(['Drag: ' num2str(mean(drag_number))])
%disp(['Acc: ' num2str(mean(a))])
%disp(['Offset: ' num2str(mean(o))])


% figure()
% plot(data1.time, data1.rpm_filtered)
% figure()
% plot(data2.time, data2.rpm_filtered)

figure()
plot(data1.time, data1.rpm_filtered, 'b', 'LineWidth', 3);

figure()
plot(data1.time, data1.power_filtered, 'k', 'LineWidth', 3)

figure()
plot(data2.time, data2.rpm_filtered, 'b', 'LineWidth', 3);

figure()
plot(data2.time, data2.power_filtered, 'k', 'LineWidth', 3)


