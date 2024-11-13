function summary_file = save_data(summary_file, subject_id, trial_id, r2_norm_train,r2_norm_test,rmse_norm_train,rmse_norm_test)
% save_data, Patrick Mayerhofer, September 2021
% stores the data to the summary file that will be saved later
    if trial_id == 1
        summary_file.r2_test_matlab_model_1(subject_id) = mean(r2_norm_test);
        summary_file.r2_train__matlab_model_1(subject_id) = mean(r2_norm_train);
        summary_file.rmse_test_matlab_model_1(subject_id) = mean(rmse_norm_test);
        summary_file.rmse_train_matlab_model_1(subject_id) = mean(rmse_norm_train);

        summary_file.r2_std_test_matlab_model_1(subject_id) = std(r2_norm_test);
        summary_file.r2_std_train_matlab_model_1(subject_id) = std(r2_norm_train);
        summary_file.rmse_std_test_matlab_model_1(subject_id) = std(rmse_norm_test);
        summary_file.rmse_std_train_matlab_model_1(subject_id) = std(rmse_norm_train);      
    elseif trial_id == 2
        summary_file.r2_test_matlab_model_2(subject_id) = mean(r2_norm_test);
        summary_file.r2_train_matlab_model_2(subject_id) = mean(r2_norm_train);
        summary_file.rmse_test_matlab_model_2(subject_id) = mean(rmse_norm_test);
        summary_file.rmse_train_matlab_model_2(subject_id) = mean(rmse_norm_train);

        summary_file.r2_std_test_matlab_model_2(subject_id) = std(r2_norm_test);
        summary_file.r2_std_train_matlab_model_2(subject_id) = std(r2_norm_train);
        summary_file.rmse_std_test_matlab_model_2(subject_id) = std(rmse_norm_test);
        summary_file.rmse_std_train_matlab_model_2(subject_id) = std(rmse_norm_train);      
    elseif trial_id == 3
        summary_file.r2_test_matlab_model_3(subject_id) = mean(r2_norm_test);
        summary_file.r2_train_matlab_model_3(subject_id) = mean(r2_norm_train);
        summary_file.rmse_test_matlab_model_3(subject_id) = mean(rmse_norm_test);
        summary_file.rmse_train_matlab_model_3(subject_id) = mean(rmse_norm_train);

        summary_file.r2_std_test_matlab_model_3(subject_id) = std(r2_norm_test);
        summary_file.r2_std_train_matlab_model_3(subject_id) = std(r2_norm_train);
        summary_file.rmse_std_test_matlab_model_3(subject_id) = std(rmse_norm_test);
        summary_file.rmse_std_train_matlab_model_3(subject_id) = std(rmse_norm_train);      
    end
end

