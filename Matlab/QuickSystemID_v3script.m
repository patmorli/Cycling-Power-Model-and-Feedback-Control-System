

%% 
for i = 1:size(powertrialsNorm,1)
    line = iddata(torquetrialsNorm(i,:)',bpmtrialsNorm(i,:)',1);
    eval(['data' num2str(i) '= line']);
    if i == 1
        together = 'data1';
    else
        together = strcat(together, ',', ['data' num2str(i)]);
    end 
end

%% I have to find a dynamical way for that... 
data = merge(data1, data2, data3, data4, data5, data6, data7, data8, data9,...
    data10, data11, data12, data13, data14, data15, data16, data17, ...
    data18, data19, data20, data21, data22);

opt = procestOptions;
opt.Display = 'on';
opt.SearchOption.MaxIter = 100;
sysalldataprocest2 = procest(data,'P2D',opt)
%sysalldatatfest1 = tfest(data,1)

%%
figure
compare(data,sys);
hold on;
plot(mean(powertrialsNorm),'r');
plot(mean(bpmtrialsNorm), '--b');

%%
sys2 = pem(data,sys)
%figure
%compare(data,sys2);


%%
d1 = iddata(power',rpm'.^2,1)
d1 = iddata(power',bpm.^2',1)

opt = procestOptions;
opt.Display = 'on';
opt.SearchOption.MaxIter = 100;
opt.InputOffset = 'auto';
opt.InitialCondition = 'auto';
opt.SearchOptions.Tolerance = 1e-6;
% opt.InputOffset = 'estimate';
% [s1,offset] = procest(d1,'P1ID',opt)
% [s1] = procest(d1,'P1D',opt)
[s1] = procest(d1,'P1D',opt)
%%
figure(1); clf;
compare(d1,s1);
%%
s2 = pem(d1,s1)
%%
figure(2); clf
[ys2,fit,x0] = compare(d1,s2);
ys2data = get(ys2);


%% 
sumnom = 0;
sumden = 0;
m = mean(power);
for i = 1:length(power)
    sumnom = sumnom + (power(i)-ys2data.OutputData{1,1}(i))^2;
    sumden = sumden + (power(i)-m)^2;
end

fitpercent = (1 - sqrt(sumnom/sumden))*100;
   





