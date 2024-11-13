function [rsquared, rsquared_alternative,mse, rmse, nrmse, norm_mean_err] = cost(x,f)
% x is the measured data set
% f is the modelled data set
n = length(x);

ssr = sum((x-f).^2);       %residual sum of squares
sst = sum((x-mean(x)).^2); %total sum of squares
rsquared = 1-(ssr/sst); 
rsquared_alternative = 1 - (ssr/sum(x.^2));
mse = mean((x-f).^2);       %mean square error
rmse = sqrt(mse);                %root mean square error
nrmse = rmse/mean(x); %normalized root mean square error
mean_err = sum(x-f)/n;
norm_mean_err = mean_err/mean(x);


end

