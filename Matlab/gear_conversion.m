function [local_summary_file, n_teeth] = gear_conversion(local_summary_file, which_bracket_from_top, trial_id, subject_id)
% which_bracket_from_top is a number between 1 and 10. Basically the
% biggest bracket is 1 and the smallest is 10.
% gear_conversion converts it to the number of teeth the bracket has

    %% if trial_id is 2, default gear stays same. 1 is one down (down one bracket, up one number), and 3 is one up
    if trial_id == 1
        which_bracket_from_top = which_bracket_from_top + 1;
    elseif trial_id == 2
        which_bracket_from_top = which_bracket_from_top - 1;
    end

    if which_bracket_from_top == 1
        n_teeth = 27;
    elseif which_bracket_from_top == 2
        n_teeth = 24;
    elseif which_bracket_from_top == 3
        n_teeth = 21; 
    elseif which_bracket_from_top == 4
        n_teeth = 19;
    elseif which_bracket_from_top == 5
        n_teeth = 17;
    elseif which_bracket_from_top == 6
        n_teeth = 16;
    elseif which_bracket_from_top == 7
        n_teeth = 15;
    elseif which_bracket_from_top == 8
        n_teeth = 14;
    elseif which_bracket_from_top == 9
        n_teeth = 13;
    elseif which_bracket_from_top == 10
        n_teeth = 12;
    end

    if trial_id == 1
        local_summary_file.rbracket_1(subject_id) = n_teeth;
    elseif trial_id == 2
        local_summary_file.rbracket_2(subject_id) = n_teeth;
    elseif trial_id == 3
        local_summary_file.rbracket_3(subject_id) = n_teeth;
    end

end

