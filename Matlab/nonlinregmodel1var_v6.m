% nonlinear regression model, find one unknown variable
% with data of whole trial
% Nov 13, 2018
%modifications: May 2018

% modifed by PK may 6th 2019 [ Now uses a contrained optimzation to fit the
% data] 

clear all;
close all

%% load a file
filename = 'Oct19OL_Patv2.mat';
load(filename);

%% define variables
c1 = 1.5; %Drag for initial guess
fbracket = 39;
rbracket = 19;
GR = fbracket/rbracket;

rw=0.3; %radius wheel
lca = 0.17; %length crank arm
m = 74;
t = (time/1000)';

x0 = 0.9; %Drag for initial guess

% % cut data to (100:end-100)
% rpmcut = rpm(100:end-100);
% powercut = power(100:end-100);
% tcut = t(100:end-100);


% cut data to (100:end-100)
rpmcut = rpm(85:2830);
powercut = power(85:2830);
tcut = t(85:2830);
bpmcut = bpm(85:2830);

% calculated speed from gear ratio and rpm
vcalc = (rw*2*pi*GR.*rpmcut)/60;



%% filter data
fs = 3;
% fs = 10;
[b,a] = butter(4, .08*2/fs);
vfilt = filtfilt(b,a,vcalc);
Pfilt = filtfilt(b,a,powercut);

%% plot data of optimization
figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power and RPM of data used for optimization')
xlabel('Trialtime [s]') 

hold on;
yyaxis left; plot(tcut, rpmcut, 'b', 'LineWidth', 3);
ylabel('Cadence [RPM]');
yyaxis right; plot(tcut, Pfilt, 'R', 'LineWidth', 3);
ylabel('Power [W]');
legend({'Cadence','Power'}, 'FontSize', 16);


%% derive
dv = nanmean([diff([vfilt NaN]); diff([NaN vfilt])]); % central difference
dt = nanmean([diff([tcut NaN]); diff([NaN tcut])]);
vdot = dv./dt;
% filter vdot
fsdot = 3;
% fs = 10;
[bdot,adot] = butter(4, .05*2/fsdot);
vdotfilt = filtfilt(bdot,adot,vdot);



figure(100); clf
    subplot(311)
        plot(tcut,vcalc, 'y'); hold on
        plot(tcut,vfilt);
    subplot(312)
%         plot(tcut,vcalc,'y'); hold on
        plot(tcut,vdot); hold on
        plot(tcut,vdotfilt);
    subplot(313)
        plot(tcut,powercut, 'y'); hold on
        plot(tcut,Pfilt);
if 1
%% SET UP YOUR problem  
tic
% parameters you defined above 
parms.m = m;
parms.rw=rw;
parms.lca = lca;
parms.vdot = vdot;
parms.GR = GR;
parms.m = m;

lb =[-1]; % Lower Bound [] 
ub =[10]; %  Upper Bound []

% setup grid of initial guesses 
inc= [0.01]; % Increments for variables to increase by from lb to ub range 
[X] = ndgrid(lb(1):inc(1):ub(1));
W = [X(:)];
%W = W + rand(size(W)); % add random noise 
W = W + -1 + 2.*rand(size(W)); % add random noise (rand from -1 to 1)
%% Setup Optimization Problem 
options = optimoptions(@fmincon,'MaxIterations',5000, 'ConstraintTolerance',1e-12,...
'OptimalityTolerance',1e-12,'FunctionTolerance',1e-12,'StepTolerance',1e-12, ...
'Algorithm','interior-point','MaxFunctionEvaluations',15000,...
'Display','final','UseParallel',false); % change these options for what you wa t 

objfun=@(x0) modelfun2(vfilt,parms,Pfilt,x0); % Objective Function 
problem = createOptimProblem('fmincon','objective',objfun,'x0',x0,'lb',lb,'ub',ub);
ms = MultiStart('PlotFcns',[]);
ms.MaxTime = 100; % Maximum Time this can evalute for.
ms.StartPointsToRun = 'all';
ms.Display = ('final');

tpoints = CustomStartPointSet(W); % set up the grid of guesses 
tpts = list(tpoints);

[c1newopt,fval,exitflag,output,solutions]= run(ms,problem,tpoints);

if 0 % you can use this functuon to plot the histogram of solutions 
Opt_Solutions_Hista2(solutions,W,1);
end

toc
end



%% regression calculations
if 1
modelfun = @(u,vfilt)(rw/lca)*GR*m*vdotfilt.*vfilt + (rw/lca)*GR*u(1)*vfilt.^3;
u = [c1]; %unknown
opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
opts.Display = 'iter';
 [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(vfilt,Pfilt,modelfun,u, opts);
 c1newreg = result;
end

%% look at accuracy of models
% %look at whole data, independent from data used for model
% rpmcut2 = rpm(100:end-100);
% powercut2 = power(100:end-100);
% tcut2 = t(100:end-100);
% vcalc2 = (rw * 2*pi*GR.*rpmcut2)/60;
% bpmcut2 = bpm(100:end-100);

rpmcut2 = rpm(85:2830);
bpmcut2 = bpm(85:2830);
powercut2 = power(85:2830);
tcut2 = t(85:2830);
vcalc2 = (rw * 2*pi*GR.*rpmcut2)/60;

% filter
v2filt = filtfilt(b,a,vcalc2);
P2filt = filtfilt(b,a,powercut2);
rpmcut2filt = filtfilt(b,a,rpmcut2)
dv2 = nanmean([diff([v2filt NaN]); diff([NaN v2filt])]); % central difference
dt2 = nanmean([diff([tcut2 NaN]); diff([NaN tcut2])]);
vdot2 = dv2./dt2;

% filter vdot2
vdot2filt = filtfilt(bdot,adot,vdot2);



 %c1newreg = 0.3597;
% fill in calculated variable 
Pnewopt = (rw/lca)*GR*m*vdot2filt.*v2filt + (rw/lca)*GR*c1newopt*v2filt.^3; 
Pnewreg = (rw/lca)*GR*m*vdot2filt.*v2filt + (rw/lca)*GR*c1newreg*v2filt.^3; 


% stats
[rsquaredreg,msereg, rmsereg] = cost(P2filt,Pnewreg)
[rsquaredreg2 rmsereg2] = rsquare(P2filt,Pnewreg)
[rsquaredopt,mseopt, rmseopt] = cost(P2filt,Pnewopt)
meanpercentagepacingerroropt2=mean(abs(P2filt-Pnewopt)./P2filt);
stdpercentagepacingerroropt2=std(abs(P2filt-Pnewopt)./P2filt);
meanpercentagepacingerrorreg2=mean(abs(P2filt-Pnewreg)./P2filt);
stdpercentagepacingerrorreg2=std(abs(P2filt-Pnewreg)./P2filt);

%% plots 
% figure;
% yyaxis left; hold on; plot(t(1305:2513), P(1305:2513), 'LineWidth', 2); plot(t(1305:2513), Pnew(1305:2513), 'LineWidth', 2); 
% ax = gca;
% %ax.LabelFontSizeMultiplier = 3;
% %ax.TitleFontSizeMultiplier = 3;
% set(gca,'FontSize',20)
% title('')
% xlabel('Trialtime [s]') ;
% ylabel('Power [W]'); 
% yyaxis right; plot(t(1305:2513), bpm(1305:2513), 'LineWidth', 2);
% plot(t(1305:2513), rpm(1305:2513), 'LineWidth', 2);
% ylabel('Cadence [bpm,rpm]') 
% legend({'Measured power','Simulated power','BPM'}, 'FontSize', 16);

figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Comparison Derived Power and Actual Power-opt')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(tcut2, Pnewopt, 'r', 'LineWidth', 3);
plot(tcut2, P2filt, 'b', 'LineWidth', 3);
legend({'Simulated','Measured'}, 'FontSize', 16);

figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Comparison Derived Power and Actual Power-reg')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(tcut2, Pnewreg, 'r', 'LineWidth', 3);
plot(tcut2, P2filt, 'b', 'LineWidth', 3);
legend({'Simulated','Measured'}, 'FontSize', 16);
%%
figure(4);
subplot(2,1,1);
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Cadence', 'fontweight', 'bold')
plot(tcut2(1:1425)-tcut2(1), rpmcut2filt(1:1425)); plot(tcut2(1:1425)-tcut2(1), bpmcut2(1:1425)); 
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');
hold off;

subplot(2,1,2); 
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power', 'fontweight', 'bold')
plot(tcut2(1:1425)-tcut2(1), P2filt(1:1425));
plot(tcut2(1:1425)-tcut2(1), Pnewreg(1:1425))
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');
hold off;
save(filename);

function Cost = modelfun2(vfilt,parms,Pfilt,x0)

vdot = parms.vdot;
rw = parms.rw;
lca = parms.lca;
GR = parms.GR;
m = parms.m;

ModelP = (rw/lca)*GR*m*vdot.*vfilt + (rw/lca)*GR*x0(1)*vfilt.^3;
R1= ModelP-Pfilt; 
%Cost= norm(R1,2).^2; %MSE?
Cost=mean(abs(R1)./Pfilt);

end

