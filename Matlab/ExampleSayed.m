
clear; clf; 

%% FILL THIS IN 

x0 = []; % Set initial arbitrary guess if good guess is unknown
lb = []; % lower bounds of possible outcomes 
ub = []; % upper bounds of possible outcomes 

% objfun=@(x0)... % put your objective function here 
% You can also choose to add a non linear contraint function if so you
% would decalre that here, 
%nonlin = @(x)..

%% Create Problem Structure 
options = optimoptions(@fmincon,'MaxIterations',5000, 'ConstraintTolerance',1e-6,...
'OptimalityTolerance',1e-6,'FunctionTolerance',1e-6,'StepTolerance',1e-6, ...
'Algorithm','interior-point','MaxFunctionEvaluations',15000,...
'Display','final','UseParallel',true); % options , using parallel computing 

problem = createOptimProblem('fmincon','objective',objfun,'x0',x0,'lb',lb,'ub',ub,...
         'options',options); % With out nonlinear contraint function 
     
% problem = createOptimProblem('fmincon','objective',objfun,'x0',x0,'lb',lb,'ub',ub,...
%          'nonlcon',constfun,'options',options);   With  nonlinear contraint function 
    

%% 
% EXAMPLE of a grid i used 

% this is optional but you can make a grid of initial guesses for your system to start from 
% a large amount of spread of places and see where you converge too. Gives
% you an idea of the basin of attraction..

if 0
numpointsIC = 50; % How many IC do you want at the end ? 
    
lb =[0.1,0.2,0.1,1.1]; % Lower Bound [Xopt, Vmax, C, Fmax] 
ub =[5,100,5,100]; %  Upper Bound [Xopt, Vmax, C, Fmax] Sampling 

inc= [.4 , 30 ,.4, 5]; % Increments for variables to increase by from lb to ub range 
[X,Y,Z,T] = ndgrid(lb(1):inc(1):ub(1),lb(2):inc(2):ub(2),lb(3):inc(3):ub(3),lb(4):inc(4):ub(4));

W = [X(:),Y(:),Z(:),T(:)];
W = W + rand(size(W));

for ik = 1:length(W)
 fvalZ(ik) =  objfun(W(ik,:)); % Heee what i did is evalaute my objective function to see if the random guess finds any sort of soultion 
end

[sorted_x, index]=sort (fvalZ,2, 'ascend'); % I then ordered the solutions in ascending order  
[~,indmax] = max(sorted_x); %sort them 
range = floor(indmax/numpointsIC); 

initial = W(index(1:numpointsIC),:); % Take 50 spread out guess that are 'good' This will be used as my guesses for Muti Start
end
%% Run MultiStart

% Option 1 using the grid generated above 

ms = MultiStart;
ms.MaxTime = 100; % Maximum Time this can evalute for you can set this for longer of course 
ms.Display = ('final');
ms.StartPointsToRun = 'all';
ms.UseParallel = true ; % This is faster if you use parallel computing FYI but you dont have too. 

tpoints = CustomStartPointSet(initial); % Setup your initial guesses 
tpts = list(tpoints);

[sol,fval,exitflag,output,solutions] = run(ms,problem,tpoints);

% Option 2 let mutlistart take some guess I find this doesnt work as well 
[sol,fval,exitflag,output,solutions] = run(ms,problem) ; % ie no tpoints 
