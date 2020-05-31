    %  This matlab program performs the value function iteration for 
%  social planner's problem in the neo-classical growth model.
%  Serdar Ozkan, ECO 2061, Winter 2016, serdar.ozkan@utoronto.ca

%Question 3
clear
clc
cd('/Users/justinleo/Documents/')

%Setting the parameter values

iter_max=160; % Choose your maximum number of iteration to make sure that V converges
%Parameter Values
alpha=0.4;
beta=0.987;
delta=0.012;

type Zprob.csv
A = readmatrix('Zprob.xlsx')
%A(:,1)

k_bar=(alpha*beta)^(1/(1-alpha))
k_bar=(alpha)^(1/(1-alpha))
% Make sure that your grid points for k include the steady state value of k
K=[0.05:0.025:0.3]; 
% K=[0.05:0.01:0.50]; % This is a finer grid points.
[m,N]=size(K);
V(1,:)=zeros(1,N); % This is my initial guess. 

for t=2:iter_max
    for i=1:N
        vmax=-100000000;
        for j=1:N
            W(t,i,j)=log(A(i,j)*K(i)^alpha-K(j) + (1-delta)*K(i))+beta*V(t-1,j);
            if(W(t,i,j)>vmax)
                vmax=W(t,i,j);
                g(t,i)=j; % Policy function
                V(t,i)=vmax; % Value function
            end
        end
    end
end


for t=2:iter_max % iterations
    for i=1:N % loop through values, k
        s = size(V(t-1,i,:));
        for j=1:length(A) % loop through states, Z 
            vmax=-100000000;
            for p=1:N % loop through optimizing variable, k'
                if A(i,j)*K(i)^alpha-K(p)+(1-delta)*K(i)<=0
                    W(t,i,j,p)=-100000000;
                else
                    W(t,i,j,p)=log(A(i,j)*(K(i)^alpha)-K(p)+(1-delta)*K(i))+...
                    beta*(dot(A(i,j,:),reshape(V(t-1,p,:), [s(2:end) 1])));
                    if W(t,i,j,p) > vmax
                        vmax = W(t,i,j,p);
                        V(t,i,j) = vmax; 
                        gk(t,i,j) = p; 
                    end
                end
            end
        end
    end
end

for t=2:iter_max % iterations
    for i=1:N % loop through values, k
        s = size(V(t-1,i,:));
        for j=1:width(Zprob) % loop through states, Z 
            vmax=-100000000;
            for p=1:N % loop through optimizing variable, k'
                if Zprob(i,j)*K(i)^alpha-K(p)+(1-delta)*K(i)<=0
                    W(t,i,j,p)=-100000000;
                else
                    W(t,i,j,p)=log(Zprob(i,j)*(K(i)^alpha)-K(p)+(1-delta)*K(i))+...
                    beta*(dot(Zprob(i,j),reshape(V(t-1,p,:), [s(2:end) 1])));
                    if W(t,i,j,p) > vmax
                        vmax = W(t,i,j,p);
                        V(t,i,j) = vmax; 
                        gk(t,i,j) = p; 
                    end
                end
            end
        end
    end
end



%Question 5
%Re-solving for delta = 1
%Solving this question is done by the same process as preceding questions

beta = 0.987;
alpha = 0.4;
ro = 0.95;
sigma = 0.007;
delta = 1;
% z_import = importdata('Z.txt');
% P_import = importdata('Zprob.txt');
% 
% z = z_import.data;
% P = P_import.data;

%%%%%%% Create the K grid %%%%%%%%%%%%%

Kss = (alpha/((1/beta)-(1-delta)))^(1/(1-alpha));   % Steady state capital

Nk=100;                                             % Number of grid points

cap_bounds=0.5;                                     % grid is 50% higher and lower than Kss

Ku=(1+cap_bounds)*Kss;                              % Upper and lower bound on capital
Kl=(1-cap_bounds)*Kss;

grid=(Ku-Kl)/(Nk-1);                                % Distance between grid points

k=Kl+grid*((1:Nk)-1);                               % Create capital grid
[h,N]=size(k);
check=150;
Vg=zeros(11,N);

while check > 0.01
    for m = 1:11
        for i =1:N
            temp=zeros(1,N);
            for j =1:N
                if z(m)*k(i)^(alpha) - k(j) + (1-delta)*k(i) < 0
                    temp(j) = -inf;
                else
                    temp(j) = log(z(m)*k(i)^alpha - k(j) + (1-delta)*k(i)) + beta*(Z_prob(m,:).*Vg(:,j));
                end
            end
        [A,B] = max(temp);
        Vu(m,i) = A;
        g(m,i)=B;
        end
    end
    check=sum(abs(Vg-Vu),'all');
    Vg=Vu;
end

%policy function guess follows from the form given in the question
%as an answer to the last question 

policy_fn_guess = alpha*beta*0.934957*k.^(alpha);
figure(4) 

% Plotting value functions for each value of z

hold off
plot(k,Vu(1,:),'+-')
hold on
plot(k,Vu(1,:),'+-')
plot(k,Vu(2,:),'+-')
plot(k,Vu(3,:),'+-')
plot(k,Vu(4,:),'+-')
plot(k,Vu(5,:),'+-')
plot(k,Vu(6,:),'+-')
plot(k,Vu(7,:),'+-')
plot(k,Vu(8,:),'+-')
plot(k,Vu(9,:),'+-')
plot(k,Vu(10,:),'+-')
plot(k,Vu(11,:),'+-')
grid on
xlabel('$k$')
ylabel('$V(k)$')
title('Value Function Iteration (delta = 1)')
leg=legend('z_1','z_2','z_3','z_4','z_5','z_6','z_7','z_8','z_9','z_10','z_11');
axis tight

figure(5) 

% Plotting policy function

hold off
plot(k,k(g(1,:)),'+-')
hold on

%the policy function guess is what we are given in Question 5

plot(k,policy_fn_guess,'+-')
grid on
xlabel('$k$')
ylabel('$g(k)$')
title('Policy Function (delta = 1)')
leg=legend('g(k)','guess');
axis tight

%Question 6

beta = 0.987;
alpha = 0.4;
ro = 0.95;
sigma = 0.007;
delta = 1;
% z_import = importdata('Z.txt');
% P_import = importdata('Zprob.txt');
% 
% z = z_import.data;
% P = P_import.data;

%generating all the steady state values

Kss = (alpha/((1/beta)-(1-delta)))^(1/(1-alpha));   % Steady state capital

Yss = Kss^(alpha)
Iss = Kss - (1-delta)*Kss
Css = Yss -Iss

%creating a k vector where the shock occurs in period 0. k(1) corresponds
%to time -1 and k(2) corresponds to time 0.

k(1) = Kss
k(2) = 1.01*Kss

%calculating the percentage change from the steady state value for captial
for t = 1:9
k(t+2) =((0.3948))*((k(t+1))^(alpha))
percent_deviation_capital(t+2) = (k(t+2) - Kss)/Kss
end

percent_deviation_capital(1) = 0
percent_deviation_capital(2) = 0.01

%finding the values for the other variable

output = k.^(alpha)
investment = (0.3948).*((k).^(alpha)) - (1-delta).*k
consumption = output - investment
output(1) = Yss
investment(1) = Iss
consumption(1) = Css

%finding the percent deviation for the other variables

for t = 1:11
percent_deviation_output(t) = (output(t)-Yss)/Yss
percent_deviation_investment(t) = (investment(t)-Iss)/Iss
percent_deviation_consumption(t) = (consumption(t)-Css)/Css
end

%plotting the IRP for periods -1 to 9

time = [-1:1:9]
figure(5) 
hold off
plot(time,percent_deviation_capital,'+-')
hold on
plot(time,percent_deviation_output,'+-')
hold on
plot(time,percent_deviation_investment,'+-')
hold on
plot(time,percent_deviation_consumption,'+-')
grid on
xlabel('Time')
ylabel('Percent Deviation from Steady State')
title('Impulse Response Function')
leg=legend('capital','output','investment','consumption');
axis tight





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%7%%%%%%%%% where Justin has made changes

%Question 7
clear

beta = 0.987;
alpha = 0.4;
ro = 0.95;
sigma = 0.007;
delta = 0.012;

% z_import = importdata('Z.txt');
% P_import = importdata('Zprob.txt');
% 
% z = z_import.data;
% P = P_import.data;

type Z.xlsx
Z.data = readmatrix('Z.xlsx')
Z.data = Z.data(:,2:end);

type Zprob.xlsx
Z_prob.data = readmatrix('Zprob.xlsx')
Z_prob.data = Z_prob.data(:,2:end);

Kss = (alpha/((1/beta)-(1-delta)))^(1/(1-alpha));   % Steady state capital

Mdl = arima('AR',{ro},'Variance',sigma,'Constant',0);
Y = simulate(Mdl,400);
figure
plot(Y)
xlim([0,400])
title('Simulated AR(1) Process')

tech_shocks = Y'

k(1) = Kss
for i=2:400
k(i) = (alpha)*(beta)*tech_shocks(i)*k(i-1)
end

time = (1:1:400)

beta = 0.987;
alpha = 0.4;
ro = 0.95;
sigma = 0.007;
delta = 0.012;
t= 400

shock = zeros(1,400)
%create shock to capital
for x = 1:400
    index = z(shock(x-1),:)
end

for i=2:400
    k(i) = (alpha)*(beta)*chain(i)*k(i-1)
    yy(i) = (k(i)^alpha)*chain(i)
    c(i) = yy(i)-k(i) + (1 - delta)*k(i-1)
    i(i) = alpha*beta*z*(yy(i))
end


for t = 1:400
    k_change(t) = (k(t)-kss)/kss
   i_change(t) = (i(t)-iss)/iss
    c_change(t) = (c(t)-css)/css
    yy_change(t) = (yy(t)-yyss)/yyss
end

kstd = std(k_change)
yystd = std(yy_change)
cstd= std(c_change)
istd= std(i_change)





figure(6) % Plotting policy function

hold off
plot(time(2:400),k(2:400),'+-')

%Doing the Value Function Iteration for the Full RBC Model
% clear
% clc
beta = 0.987;
alpha = 0.4;
ro = 0.95;
sigma = 0.007;
delta = 0.012;
b=2;

%%%%%%% Create the K grid %%%%%%%%%%%%%

Kss = (alpha/((1/beta)-(1-delta)))^(1/(1-alpha));   % Steady state capital

Nk=25;                                             % Number of grid points

cap_upper_bounds=0.5;                                     % grid is 50% higher and lower than Kss
cap_lower_bounds=0.5;

Ku=(1+cap_upper_bounds)*Kss;                              % Upper and lower bound on capital
Kl=(1-cap_lower_bounds)*Kss;

grid=(Ku-Kl)/(Nk-1);                                % Distance between grid points

k=Kl+grid*((1:Nk)-1);                                  % Create capital grid
l = [0.01:0.01:0.25];
[h,N]=size(k);
check=100;
Vg=zeros(11,N);

Z = cell2mat(struct2cell(Z));
Z_prob = cell2mat(struct2cell(Z_prob));


while check > 0.01
    for m = 1:11
        for i =1:N
            temp=zeros(1,N);
            for j =1:N
            temp1=zeros(1,N);
                for u = 1:N
                    if Z(m)*(k(i)^(alpha))*(l(u)^(1-alpha)) - k(j) + (1-delta)*k(i) < 0
                        temp(u) = -inf; 
                    else
                       temp(1:10) = log(Z(m)*(k(i)^(alpha))*(l(u)^(1-alpha)) - k(j) + (1-delta)*k(i)) + b*log(1-l(u)) + beta*(Z_prob(m,:).*Vg(:,m));
                    end %%% error is in above line %%%
                end 
%                 
                [C,D] = max(temp); %max(temp)
                optimal_labor_value(m,j) = l(4); 
                if Z(m)*(k(i)^(alpha))*(l(u)^(1-alpha)) - k(j) + (1-delta)*k(i) < 0
                        temp1(j) = -inf;
                else 

                      temp1(1:10) = log(Z(m)*(k(i)^(alpha))*(l(u)^(1-alpha)) - k(j) + (1-delta)*k(i)) + b.*log(1-optimal_labor_value(m,j)) + beta*(Z_prob(m,:).*Vg(:,m));
               end
            end
%             
        [A,B] = max(temp1);
        Vu(m,i) = A;
        g(m,i)=B;
        q(m,i)=D;  
        end
    check=sum(abs(Vg-Vu),'all');
    Vg=Vu;
    end
end

figure(6) % Plotting value functions for each value of z
hold off
plot(k,Vu(1,:),'+-')
hold on
plot(k,Vu(1,:),'+-')
plot(k,Vu(2,:),'+-')
plot(k,Vu(3,:),'+-')
plot(k,Vu(4,:),'+-')
plot(k,Vu(5,:),'+-')
plot(k,Vu(6,:),'+-')
plot(k,Vu(7,:),'+-')
plot(k,Vu(8,:),'+-')
plot(k,Vu(9,:),'+-')
plot(k,Vu(10,:),'+-')
plot(k,Vu(11,:),'+-')
grid on
xlabel('$k$')
ylabel('$V(k)$')
title('Value Function Iteration (delta = 1)')
leg=legend('z_1','z_2','z_3','z_4','z_5','z_6','z_7','z_8','z_9','z_10','z_11');
axis tight

figure(7) % Plotting policy function for capital
hold off
plot(k,k(g(1,:)),'+-')
hold on
plot(k,policy_fn_guess,'+-')
grid on
xlabel('$k$')
ylabel('$g(k)$')
title('Policy Function (delta = 1)')
leg=legend('g(k)','guess');
axis tight

figure(7) % Plotting policy function for labor
hold off
plot(k,k(g(1,:)),'+-')
hold on
plot(k,policy_fn_guess,'+-')
grid on
xlabel('$k$')
ylabel('$g(k)$')
title('Policy Function (delta = 1)')
leg=legend('g(k)','guess');
axis tight

for t = 1:9
k(t+2) =((alpha^b))*((k(t+1))^(alpha))
percent_deviation_capital(t+2) = (k(t+2) - Kss)/Kss
end

percent_deviation_capital(1) = 0
percent_deviation_capital(2) = 0.01

%finding the values for the other variable

output = k.^(alpha)
investment = (alpha^b).*((k).^(alpha)) - (1-delta).*k
consumption = output - investment
output(1) = Yss
investment(1) = Iss
consumption(1) = Css

%finding the percent deviation for the other variables

for t = 1:11
percent_deviation_output(t) = (output(t)-Yss)/Yss
percent_deviation_investment(t) = (investment(t)-Iss)/Iss
percent_deviation_consumption(t) = (consumption(t)-Css)/Css
end

std(percent_deviation_output(t))
std(percent_deviation_investment(t))
std(percent_deviation_consumption(t))

beta = 0.987;
alpha = 0.4;
ro = 0.95;
sigma = 0.007;
delta = 0.012;
b=2;

%% GIVEN K & Z, TRY TO OPTIMIZE K' & L
Kss=0
l = 0
l_grid = linspace(0,2,100)
for y=2:20;
    for c=1:20;
        l = ((y/2*c)*(1-alpha))/(1+(1-alpha)*(y/2*c))
        
        
        if (c/y) == 1-alpha;
        %if l==(1/3);    
            vmax=1000;
            Kss= (((alpha/(1/beta)-(1-delta))).^(1/(1-alpha)))*(l);
            %Kss= (((1/beta)-1+delta)/(alpha)).^(1/(1-alpha))*l;
        else

            Kss=0
            
        end
        max(Kss)        
        
    end
end

fun = @(x)3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2);
x0 = [1,1];
[x,fval] = fminunc(fun,x0)
