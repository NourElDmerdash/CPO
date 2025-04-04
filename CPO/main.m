%_________________________________________________________________________%
%  Crested Porcupine Optimizer: A new nature-inspired metaheuristic               %
%                                                                         %
%  Developed in MATLAB R2019A                                      %
%                                                                         %
%  Author and programmer: Reda Mohamed (E-mail: redamoh@zu.edu.eg) & Mohamed Abdel-Basset (E-mail: mohamedbasset@ieee.org)                              %
%                                                                         %
%   Main paper: Abdel-Basset, M., Mohamed, R.                                    %
%               Crested Porcupine Optimizer: A new nature-inspired metaheuristic              %
%                  %
%                                                                         %
%%
clear all
clc
CP_no=130; % Number of search agents (crested porcupines)
Tmax=1000000; % Maximum number of Function evaluations
RUN_NO=30; %% Number of independent runs
Fun_id=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30];

fhd=str2func('cec17_func'); %%Default Benchmark
cec=2;
if cec==1 %% CEC-2014
    fhd=str2func('cec14_func');
    benchmarksType='cec14_func';
elseif cec==2 %% CEC-2017
    fhd=str2func('cec17_func');
    benchmarksType='cec17_func';
elseif cec==3 %% CEC-2020
    fhd=str2func('cec20_func');
    benchmarksType='cec20_func';
elseif cec==4 %% CEC-2022
    fhd=str2func('cec22_func');
    benchmarksType='cec22_func';
end

for i=1:30
 tic;  
 if cec==2 && i==2
       continue;
 elseif cec==3 && i>10
       return
 elseif cec==4 && i>12
       return
 end
 parfor j=1:RUN_NO
  [lb,ub,dim]=Get_Functions_detailsCEC(Fun_id(i));
  fobj=Fun_id(i);
  [Best_score,Best_pos,Convergence_curve(j,:)]=CPO(CP_no,Tmax,ub,lb,dim,fobj,fhd);
  fitness(1,j)=Best_score;
 end
time=toc;
fprintf(['benchmark   \t',num2str(cec),'\t','Function_ID\t',num2str(i),'\tAverage Fitness:',num2str(mean(fitness(1,:)),20),'\tAverage Time:',num2str(time/RUN_NO,5),'\n']);
%% Drawing Convergence Curve %%
figure(i)
h=semilogy(mean(Convergence_curve),'-<','MarkerSize',8,'LineWidth',1.5);
h.MarkerIndices = 1000:20000:Tmax;
xlabel('Function Evaluation');
ylabel('Best Fitness obtained so-far');
axis tight
grid off
box on
legend({'CPO'});
clear Convergence_curve;
end