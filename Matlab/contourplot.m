%% contourplot with specific heights, Patrick Mayerhofer, October 2019

figure;
[X1 Y1] = meshgrid(Kp_vals,Ki_vals);
% [X1,Y1,MSE] = peaks;
%best visualization: 
contour(X1,Y1,MSE, [900:200:5000],'ShowText', 'On' )
hold on;