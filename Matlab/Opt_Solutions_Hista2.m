function [outputArg1] = Opt_Solutions_Hista2(solutions,initial,fig)

value = [];
index = 1; 
DetA = size(initial);
col = DetA(2); 

for i = 1:length([solutions.Fval]) 
    store(i) = length([solutions(1,i).X0]);
    value = store(i); 
    if i==1
    outputArg1.parms= zeros(store, col); 
    for a = 1:value
          outputArg1.parms(a,1:col) = solutions(1,i).X;  
    end
   
    
    else
        
         if col==1
    holder(1:value,col) =solutions(1,i).X(1);
    outputArg1.parms(index:index+value-1,1:col) = holder;
                
         elseif col<4 
    holder(1:value,col) =solutions(1,i).X(1);
    
    holder(1:value,1) =solutions(1,i).X(1);
    holder(1:value,2) =solutions(1,i).X(2);
    holder(1:value,3) =solutions(1,i).X(3);
    outputArg1.parms(index:index+value-1,1:col) = holder;
    
        else
    holder(1:value,col) =solutions(1,i).X(1);
    
    holder(1:value,1) =solutions(1,i).X(1);
    holder(1:value,2) =solutions(1,i).X(2);
    holder(1:value,3) =solutions(1,i).X(3);
    holder(1:value,4) =solutions(1,i).X(4);
    outputArg1.parms(index:index+value-1,1:col) = holder;
            
        end 
    end
    index = index+value; 
    holder=[];
end

value = [];
index = 1; 

for i = 1:length([solutions.Fval]) 
    value = store(i); 
    
    matrix = [solutions.Fval]; 
    if i ==1 
    outputArg1.fval(1:value) = matrix(i); 
    else
    outputArg1.fval(index:index+value-1) = matrix(i);     
    end
    
    index = index+value; 
    
end
outputArg1.fval = outputArg1.fval';

inf_count = length(initial) - sum(store);


if fig ==1 
figure()
h1 =histogram([outputArg1.fval],10); 
hold on 
h2 = line([xlim],[inf_count inf_count],'Color','r','LineWidth',5)

xlabel('Fval','FontSize', 18)
ylabel('# of unique solutions','FontSize', 18) 
set(gca,'FontSize',18)
legend([h1 h2],'# Converged Sol',' # Infeasible Sol')
end
end

%% 
