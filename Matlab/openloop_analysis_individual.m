% openloop_analysis
% September, 2021, Patrick Mayerhofer

clear all;
close all;
%% Changeable variables
% try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
s_id = [2,3,4,5,6,7,8,9,10,11,12,13];
t_id = [1,2];
normalize_flag = 1;
save_flag = 1;
plot_flag = 0;
acc_term_flag = 0;
offset_term_flag = 0;
acc_term = 0;
offset_term = 0;
c1 = 1; %initial guess
opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
old_model = 0;
bike_and_equipment_weight = 13;



%% load summary file
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/';
% excel summary file 
dir_summary_file = [dir_root, 'Summary.xlsx'];
summary_file = readtable(dir_summary_file, 'Format', 'auto');
local_summary_file = summary_file; % because the mat file also has summary file saved, and overwrites it otherwise when we load it

%% bikespecs
rw = 0.3; %works best with actual gpsspeed
%rw=0.325; %radius wheel original
lca = 0.17; %length crank arm
fbracket = 39;
drag_number = [];
for subject_id = s_id
    drag_number_subjects = [];
    m = summary_file.Weight(subject_id) + bike_and_equipment_weight;

    for trial_id = t_id

        %% rbracket and gear ratio
        if trial_id == 1
            rbracket = summary_file.rbracket_1(subject_id);
        elseif trial_id == 2
            rbracket = summary_file.rbracket_2(subject_id);
        elseif trial_id == 3
            rbracket = summary_file.rbracket_3(subject_id);
        end

        GR = fbracket/rbracket;


        %% load cycling data
        filename = ['Subject', num2str(subject_id), '_', num2str(trial_id)];
        dir_load_file = [dir_root, 'OpenLoop/CleanedCSV/', filename, '.csv'];
        data = readtable(dir_load_file);
        
        % add GR to dataset
        data.GR = zeros(size(data,1),1) + GR;

        %% put in 3 sets
        l = size(data,1);
        l3 = round(l/3);
        increments = [1, l3, l3*2, l];
        data1 = data(increments(1):increments(2), :);
        data2 = data(increments(2):increments(3), :);
        data3 = data(increments(3):increments(4), :);

        %% do each calculation with all 3 increments
        for i = 1:3
            %% get variables for calculation
            % create array that shows what should be used for test and
            % training
            ones_matrix_test = zeros(1,l);
            ones_matrix_train = ones(1,l);
            for u = increments(i):increments(i+1)
                ones_matrix_test(u) = 1;
                ones_matrix_train(u) = 0;
            end
            
            % divide in train and test data
            data_test{i} = data(logical(ones_matrix_test),:);
            data_train{i} = data(logical(ones_matrix_train),:);
            
            %v_train = data_train.gpsSpeed_filtered;
            %vdot_train = data_train.vdot_filtered;
            v_train = data_train{i}.vcalc_filtered; 
            vdot_train = data_train{i}.vcalcdot_filtered;
            P_train = data_train{i}.power_filtered;
            time_train = data_train{i}.time;
            rpm_train = data_train{i}.rpm_filtered;
            GR_train = data_train{i}.GR;
            v_vdot_GR_train = [v_train, vdot_train, GR_train];


            v_test = data_test{i}.vcalc_filtered;
            vdot_test = data_test{i}.vcalcdot_filtered;
            P_test = data_test{i}.power_filtered;
            time_test = data_test{i}.time;
            rpm_test = data_test{i}.rpm_filtered;
            GR_test = data_test{i}.GR;
            v_vdot_GR_test = [v_test,vdot_test,GR_test];

             
            %% regression start 
            if acc_term_flag == 0 & offset_term_flag == 0 % original, only the drag term
                load_and_save = '_onlydrag';
                if old_model == 0
                    modelfun = @(u,v_dot_GR_train)m.*v_vdot_GR_train(:,1).*v_vdot_GR_train(:,2) + u(1).*v_vdot_GR_train(:,1).^3;
                else
                    modelfun = @(u,v_vdot_GR_train)v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3;
                end
                u = [c1]; %unknown

                [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_GR_train, P_train,modelfun,u, opts);
                c1new = result(1);

                d(i) = c1new;

                %% predict power with optimized variable(s)
                if old_model == 0
                    Pnewreg_train{i} = m.*v_vdot_GR_train(:,1).*v_vdot_GR_train(:,2) + c1new.*v_vdot_GR_train(:,1).^3;
                    Pnewreg_test{i} = m.*v_vdot_GR_test(:,1).*v_vdot_GR_test(:,2) + c1new.*v_vdot_GR_test(:,1).^3;
                end
                if old_model
                     Pnewreg_train{i} = v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3;
                     Pnewreg_test{i} = v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3; 
                end
            end

            if acc_term_flag == 1 & offset_term_flag == 0 % drag and acceleration term c
                load_and_save = '_dragandaccterm';
                modelfun = @(u,v_dot_train)u(2)*m.*v_vdot_train(:,1).*v_vdot_train(:,2) + u(1).*v_vdot_train(:,1).^3;
                %modelfun = @(u,v_GR_train)u(2)*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3;
                u = [c1 acc_term]; %unknown

                [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_GR_train, P_train,modelfun,u, opts);
                c1new = result(1);
                acc_termnew = result(2);

                d(i) = c1new;
                a(i) = acc_termnew;

                %% predict power with optimized variable(s)
                Pnewreg_train{i} = acc_termnew*m.*v_vdot_GR_train(:,1).*v_vdot_GR_train(:,2) + c1new.*v_vdot_GR_train(:,1).^3;
                Pnewreg_test{i} = acc_termnew*m.*v_vdot_GR_test(:,1).*v_vdot_GR_test(:,2) + c1new.*v_vdot_GR_test(:,1).^3;
%                 Pnewreg_train{i} = acc_termnew*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3;
%                 Pnewreg_test{i} = acc_termnew*(rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3; 
            end

            if acc_term_flag == 1 & offset_term_flag == 1 % drag, acceleration, and offset term
                load_and_save = '_allterms';
                modelfun = @(u,v_dot_GR_train)u(2)*m.*v_vdot_GR_train(:,1).*v_vdot_GR_train(:,2) + u(1).*v_vdot_GR_train(:,1).^3 + u(3);
                %modelfun = @(u,v_GR_train)u(2)*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3 + u(3);
                u = [c1 acc_term offset_term]; %unknown

                [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_GR_train, P_train,modelfun,u, opts);
                c1new = result(1);
                acc_termnew = result(2);
                offset_termnew = result(3);

                d(i) = c1new;
                a(i) = acc_termnew;
                o(i) = offset_termnew;

                %% predict power with optimized variable(s)
                Pnewreg_train{i} = acc_termnew*m.*v_vdot_GR_train(:,1).*v_vdot_GR_train(:,2) + c1new.*v_vdot_GR_train(:,1).^3 + offset_termnew;
                Pnewreg_test{i} = acc_termnew*m.*v_vdot_GR_test(:,1).*v_vdot_GR_test(:,2) + c1new.*v_vdot_GR_test(:,1).^3 + offset_termnew;
%                 Pnewreg_train{i} = acc_termnew*(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3+offset_termnew;
%                 Pnewreg_test{i} = acc_termnew*(rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3+offset_termnew; 
            end

            if acc_term_flag == 0 & offset_term_flag == 1 % drag and offset term
                load_and_save = '_dragandoffsetterm';
                modelfun = @(u,v_dot_GR_train)m.*v_vdot_GR_train(:,1).*v_vdot_GR_train(:,2) + u(1).*v_vdot_GR_train(:,1).^3 + u(2);
                %modelfun = @(u,v_GR_train)(rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*u(1).*v_vdot_GR_train(:,1).^3 + u(2);
                u = [c1 offset_term]; %unknown

                [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v_vdot_GR_train, P_train,modelfun,u, opts);
                c1new = result(1);
                offset_termnew = result(2);

                d(i) = c1new;
                o(i) = offset_termnew;

                %% predict power with optimized variable(s)
                Pnewreg_train{i} = m.*v_vdot_GR_train(:,1).*v_vdot_GR_train(:,2) + c1new.*v_vdot_GR_train(:,1).^3 + offset_termnew;
                Pnewreg_test{i} = m.*v_vdot_GR_test(:,1).*v_vdot_GR_test(:,2) + c1new.*v_vdot_GR_test(:,1).^3 + offset_termnew;
%                 Pnewreg_train{i} = (rw/lca)*v_vdot_GR_train(:,3).*m.*v_vdot_GR_train(:,2).*v_vdot_GR_train(:,1) + (rw/lca)*v_vdot_GR_train(:,3)*c1new.*v_vdot_GR_train(:,1).^3+offset_termnew;
%                 Pnewreg_test{i} = (rw/lca)*v_vdot_GR_test(:,3).*m.*v_vdot_GR_test(:,2).*v_vdot_GR_test(:,1) + (rw/lca)*v_vdot_GR_test(:,3)*c1new.*v_vdot_GR_test(:,1).^3+offset_termnew; 
            end

            %% variance
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
                plot(Pnewreg_test{1}, 'b', 'LineWidth', 3);
                plot(P_test, 'r', 'LineWidth', 3);
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
                legend({'Predicted','Measured'}, 'FontSize', 16);
            end
        end
        
        drag_number_subjects = [d drag_number_subjects];

        %% calculate average and std too
%         local_summary_file.r2_norm_test(subject_id) = mean(r2_norm_test);
%         local_summary_file.r2_norm_train(subject_id) = mean(r2_norm_train);
%         local_summary_file.rmse_norm_test(subject_id) = mean(rmse_norm_test);
%         local_summary_file.rmse_norm_train(subject_id) = mean(rmse_norm_train);
% 
%         local_summary_file.r2_norm_test_std(subject_id) = std(r2_norm_test);
%         local_summary_file.r2_norm_train_std(subject_id) = std(r2_norm_train);
%         local_summary_file.rmse_norm_test_std(subject_id) = std(rmse_norm_test);
%         local_summary_file.rmse_norm_train_std(subject_id) = std(rmse_norm_train);
        
%         %% fill summary file and if true, save data to excel sheet
%         local_summary_file = save_data(local_summary_file, subject_id, trial_id, r2_norm_train,r2_norm_test,rmse_norm_train,rmse_norm_test);
%         if save_flag
%             
%             save([dir_root, 'OpenLoop/ResultMatFiles/', filename])
%         end
        if save_flag
            dir_save = [dir_root, 'OpenLoop/ResultMatFiles/Individual/Subject', num2str(subject_id), '_', num2str(trial_id) , load_and_save, '.mat'];
            save(dir_save, 'data_train', 'data_test', 'Pnewreg_train', 'Pnewreg_test', 'power_measured_norm_train', 'power_measured_norm_test', 'power_predicted_norm_train', 'power_predicted_norm_test', 'r2_norm_train', 'r2_norm_test', 'r2_train', 'r2_alternative_train', 'r2_test', 'r2_alternative_test', 'rmse_norm_train', 'rmse_norm_test', 'rmsereg_train', 'rmsereg_test', 'nrmse_train', 'nrmse_test', 'norm_mean_err_train', 'norm_mean_err_test');

            %writetable(local_summary_file, dir_summary_file);

        end
    end
    drag_number = [drag_number drag_number_subjects];
end


% if in a loop, do this only in the very end.

