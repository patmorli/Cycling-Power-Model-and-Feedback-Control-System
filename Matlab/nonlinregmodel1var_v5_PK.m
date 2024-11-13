% nonlinear regression model, find one unknown variable
% with data of whole trial
% Nov 13, 2018

% modifed by PK may 6th 2019 [ Now uses a contrained optimzation to fit the
% data] 

clear all;
close all

%% load a file
filename = 'CHF1';
load(filename);

%% define variables

fbracket = 39;
rbracket = 16;
GR = fbracket/rbracket;

rw=0.3344; %radius wheel
lca = 0.17; %length crank arm
m = 84.2;
t = (time/1000)';

x0 = 0.9; %Drag for initial guess
%vcalc = (0.3344 * 2*pi*GR)./(1./(rpm/60));

% cut data to (100:end-100)
rpmcut = rpm(100:end-100);
powercut = power(100:end-100);
tcut = t(100:end-100);

% calculated speed from gear ratio and rpm
vcalc = (rw * 2*pi*GR.*rpmcut)/60;

%% filter data
fs = 3;
[b,a] = butter(4, .1*2/fs);
v = filtfilt(b,a,vcalc);
P = filtfilt(b,a,powercut);

%% derive
dv = nanmean([diff([v NaN]); diff([NaN v])]); % central difference
dt = nanmean([diff([tcut NaN]); diff([NaN tcut])]);
vdot = dv./dt;

%% SET UP YOUR problem  
tic
% parameters you defined above 
parms.m = m;
parms.rw=rw;
parms.lca = lca;
parms.vdot = vdot;
parms.GR = GR;
parms.m = m;

lb =[0]; % Lower Bound [] 
ub =[10]; %  Upper Bound []

% setup grid of initial guesses 
inc= [0.01]; % Increments for variables to increase by from lb to ub range 
[X] = ndgrid(lb(1):inc(1):ub(1));
W = [X(:)];
%W = W + rand(size(W)); % add random noise 
W = W + -1 + 2.*rand(size(W));
%% Setup Optimization Problem 
options = optimoptions(@fmincon,'MaxIterations',5000, 'ConstraintTolerance',1e-12,...
'OptimalityTolerance',1e-12,'FunctionTolerance',1e-12,'StepTolerance',1e-12, ...
'Algorithm','interior-point','MaxFunctionEvaluations',15000,...
'Display','final','UseParallel',false); % change these options for what you wa t 

objfun=@(x0) modelfun2(v,parms,P,x0); % Objective Function 
problem = createOptimProblem('fmincon','objective',objfun,'x0',x0,'lb',lb,'ub',ub);
ms = MultiStart('PlotFcns',[]);
ms.MaxTime = 100; % Maximum Time this can evalute for.
ms.StartPointsToRun = 'all';
ms.Display = ('final');

tpoints = CustomStartPointSet(W); % set up the grid of guesses 
tpts = list(tpoints);

[c1new,fval,exitflag,output,solutions]= run(ms,problem,tpoints);

if 1 % you can use this functuon to plot the histogram of solutions 
Opt_Solutions_Hista2(solutions,W,1);
end

toc

%% fill in calculated variable 
Pnew = (rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*c1new*v.^3; 

%% variance
[r, P1]=corrcoef(P, Pnew);
r2_OptimizationMethod = r(2,1)^2

%% regression calculations
if 1
opts.TolFun =1e-1000;   %termination tolerance for residual sum of squares
opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';
opts.MaxIter = 1000;
 [result,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(v,P,modelfun,u, opts);
 c1new = result;
 Pnew2 = (rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*c1new*v.^3; 
 [r, P1]=corrcoef(P, Pnew2);
r2_RegressionMethod = r(2,1)^2
end

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
title('Comparison Derived Power and Actual Power')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(tcut, Pnew, 'r', 'LineWidth', 3);
plot(tcut, P, 'b', 'LineWidth', 3);
legend({'Simulated','Measured'}, 'FontSize', 16);


%%
save(filename);

function Cost = modelfun2(v,parms,P,x0)

vdot = parms.vdot;
rw = parms.rw;
lca = parms.lca;
GR = parms.GR;
m = parms.m;

ModelP = (rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*x0(1)*v.^3;
R1= ModelP-P; 
Cost= norm(R1,2).^2; % Reduce R2; 

end

