% To run only one line: <Fn + Shift + F7>
%% 1. Setup
clear all
close all
close hidden
warning off all
clc

cd '/Users/justinleo/Downloads/2404Folder'

% *************************************************************************
% Define paths for codes, optimization results and logs
% *************************************************************************
code_path    =pwd;
results_path =[code_path,'/optimization results/'];
logs_path    =[code_path,'/optimization logs/'];
add_path     =[code_path,'/optimization routines/'];
addpath(add_path);

%**************************************************************************
%Define globals
%**************************************************************************
global ns x1 x2 s_jt vfull dfull theta1 theti thetj...
       cdid cdindex IV1 invA1 nbrn...
       mvalold_logit v mvalold oldt2 gmmresid...
       mymaxfunevals mvalolds mvalold0 mvalold00 mvalolds2...
       fvals_track fcnevals ppp mval_track       

%**************************************************************************
%Load data and define various parameters, including ownership dummies
%**************************************************************************  
load BLP_data cdid cdindex share outshr price firmid id const hpwt air mpd space mpg trend...
     product model_id own_dummies
load BLP_data_str model_name 

%% 1.a)
% Code From The Main File
    X=[const,(hpwt),air,(mpg),space];
    for i=1:size(id,1)
        other_ind=(firmid==firmid(i)  & cdid==cdid(i) & id~=id(i));
        rival_ind=(firmid~=firmid(i)  & cdid==cdid(i));
        total_ind=(cdid==cdid(i));
        sum_other(i,:)=sum(X(other_ind==1,:));
        sum_rival(i,:)=sum(X(rival_ind==1,:));
        sum_total(i,:)=sum(X(total_ind==1,:));
    end
   
    %variables in demand without random coefficients
    x1 = [price,X]; clear X
    
    s_jt=share;
    y = log(s_jt) - log(outshr);


% New Code
% Coefficients
logit_coef = inv(x1'*x1)*x1'*y

% Robust Std. Errors
residuals = y - x1*logit_coef;
covmatrix = inv(x1'*x1)*(x1'*diag(residuals.^2)*x1)*inv(x1'*x1);
robust_se = sqrt(diag(covmatrix))

%% 1.b)

sum_other=[];
sum_rival=[];

X=[const,(hpwt),air,(mpg),space];
for i=1:size(id,1)
    other_ind=(firmid==firmid(i)  & cdid==cdid(i) & id~=id(i));
    rival_ind=(firmid~=firmid(i)  & cdid==cdid(i));
    total_ind=(cdid==cdid(i));
    sum_other(i,:)=sum(X(other_ind==1,:));
    sum_rival(i,:)=sum(X(rival_ind==1,:));
    sum_total(i,:)=sum(X(total_ind==1,:));
end

IV1=[X,sum_other,sum_rival];

%variables in demand without random coefficients
x1=[price,X]; clear X

s_jt=share;
y = log(s_jt) - log(outshr);

% Added Code
% Coefficients
Z = IV1*inv(IV1'*IV1)*IV1';
IV_coefficients = inv(x1'*Z*x1)*x1'*Z*y

% Robust Std. Errors
residuals_2 = y - x1*IV_coefficients;
covmatrix_2 = inv(x1'*Z*x1)*(x1'*Z*diag(residuals_2.^2)*Z*x1)*inv(x1'*Z*x1);
robust_se_2 = sqrt(diag(covmatrix_2))

%% 1.c)
% The following code is from main.m; it was altered such that the program only uses one optimization routine

for optrout=1:1  % Changed from 1:10
    
    perturbs=   (1:1:50)';
    mytolx=         1e-3;
    mytolfun=       1e-3;
    mymaxiters=   5*10^5;
    mymaxfunevals=  4000;
    
    fvals_track=NaN*ones(mymaxfunevals,size(perturbs,1));    

    if optrout<=9
       outfile=[logs_path,['blp_0',num2str(optrout),'_optim_log.txt']];
       matfile=['blp_0',num2str(optrout),'_data_optim'];
    else
       outfile=[logs_path,['blp_',num2str(optrout),'_optim_log.txt']];
       matfile=['blp_',num2str(optrout),'_data_optim'];
    end

    fid = fopen(outfile,'w'); fclose(fid);

    % *************************************************************************
    counts2    =[];                  %store function evaluations
    deltas     =[];                  %store deltas
    exit_infos =[];                  %store exit info
    fvals      =[];                  %store GMM values
    gmmresids  =[];                  %store gmm residuals
    gradients  =[];                  %store analytical gradients
    gradients2 =[];                  %store numerical  gradients I
    gradients3 =[];                  %store numerical  gradients II   
    hessians   =[];                  %store hessians
    hessians2  =[];                  %store hessians II    
    mvalolds2  =[];                  %store mvalolds
    perturbs2  =[];                  %store perturbation set number
    std_errors =[];                  %store std.errors
    theta1s    =[];                  %store theta1s
    theta2s    =[];                  %store theta2s
    fvals_track=[];                  %store GMM values in all evaluations
    tocs       =[];                  %store time of completion

    % *********************************************************************
    % Demand instruments
    % *********************************************************************
    sum_other=[];
    sum_rival=[];

    X=[const,(hpwt),air,(mpg),space];
    for i=1:size(id,1)
        other_ind=(firmid==firmid(i)  & cdid==cdid(i) & id~=id(i));
        rival_ind=(firmid~=firmid(i)  & cdid==cdid(i));
        total_ind=(cdid==cdid(i));
        sum_other(i,:)=sum(X(other_ind==1,:));
        sum_rival(i,:)=sum(X(rival_ind==1,:));
        sum_total(i,:)=sum(X(total_ind==1,:));
    end
    
    IV1=[X,sum_other,sum_rival];

    %Load N(0,I) draws
    load v

    % %ownership structure matrix
    % own=own_dummies; clear owndummies

    %variables in demand without random coefficients
    x1=[price,X]; clear X

    %variables in demand with random coefficients
    x2=x1(:,1:size(x1,2)-1);

    %#of indivduals and markets
    ns =  size(v,2)/5;
    nmkt = 20;

    %#of brands per market
    nbrn=zeros(nmkt,1);
    nbrn(1)=cdindex(1);
    for i=2:max(nmkt)
    nbrn(i)=sum(cdid==i);
    end

    %demographics and N(0,I) unobservables
    demogr=zeros(size(v));
    vfull = v(cdid,:);
    dfull = demogr(cdid,:);

    % *********************************************************************
    % Logit regressions
    % *********************************************************************
    theta2=ones(5,1);
    theta2w=zeros(5,5);
    theta2w(:,1)=theta2;
    [theti, thetj, theta2]=find(theta2w);

    invA1 = inv(IV1'*IV1);

    s_jt=share;
    y = log(s_jt) - log(outshr);

    mid = x1'*IV1*invA1*IV1';
    theta1 = inv(mid*x1)*mid*y;
    mvalold_logit = x1*theta1;

    n=size(x1,1);
    k=size(x1,2);

    ESS=y'*y-2*theta1'*x1'*y+theta1'*x1'*x1*theta1;
    s2=ESS/(n-k);
    A=(x1'*(IV1*inv(IV1'*IV1)*IV1')*x1);
    se=sqrt(diag(s2*inv(A)));

    mvalold=exp(mvalold_logit);
    oldt2=zeros(size(theta2));

    % *********************************************************************
    % Start optimization routine with perturb_no different starting values
    % for theta2: normrnd(0,1,size(theta2));
    % for delta:  delta_logit+normrnd(0,stddev(delta_logit),2256,1)
    % *********************************************************************
    diary(outfile)
    
    for ppp=1:size(perturbs,1)

        tic
        perturb=perturbs(ppp,1);
        fprintf('\n');
        fprintf('========================================================================\n')
        fprintf('                       optimization routine :%2i\n',optrout);
        fprintf('                     set of starting values :%2i\n',perturb);
        fprintf('========================================================================\n')
        theta2w=     [3.612 0 0 0 0;
                      4.628 0 0 0 0;
                      1.818 0 0 0 0;
                      1.050 0 0 0 0;
                      2.056 0 0 0 0];

        [theti, thetj, theta2]=find(theta2w);
        randn('state',1000*perturb);
        theta2=normrnd(0,1,size(theta2));
        theta2w0= full(sparse(theti,thetj,theta2));

        randn('state',1000*perturb);

        mvalold=mvalold_logit+normrnd(0,sqrt(s2),size(x1,1),1);

        %initialize matrix tracking deltas
        mval_track=[mvalold_logit];

        mvalold=exp(mvalold);
        oldt2 = zeros(size(theta2));

        fprintf('theta2 starting values:\n')        
        printm(theta2');
        
        %initialize counter of function evaluations
        fcnevals=0;

        % *****************************************************************
        % Quasi-Newton I
        % *****************************************************************
        if optrout==1
            options = optimset(...
                'GradObj',     'on',...
                'HessUpdate',  'bfgs',...
                'LargeScale',  'off',...
                'MaxFunEvals', mymaxfunevals,...
                'TolFun',      mytolfun,...
                'TolX',        mytolx,...
                'MaxIter',     mymaxiters,...
                'Display','iter');
            [theta2,fval,exit_info,tmp]=fminunc('gmmobj2',theta2,options);     
            counts=tmp.funcCount;
        end

        % *****************************************************************
        % During attempted function evaluations, deltas may be generated
        % with missing values; in this case, use the last vector of deltas
        % with non-missing values
        % *****************************************************************        
        xxx=(1:1:size(mval_track,2))';
        yyy=sum(isnan(mval_track))'==0;
        xxx=max(xxx(yyy==1,:));
        mvalold=mval_track(:,xxx);
        
        if xxx>1
            mvalolds=mval_track(:,[xxx-1,xxx]);
        end
        if xxx==1
            mvalold=mvalold_logit;            
            mvalolds=[mvalold_logit,mvalold];
        end        

        counts2    = [counts2;counts];
        perturbs2  = [perturbs2;perturb];
        gmmresids  = [gmmresids,gmmresid];
        deltas     = [deltas,log(mvalold)];
        fvals      = [fvals;fval];
        
        theta2s    = [theta2s;theta2'];

        if isempty(theta1)
            theta1=-999999*ones(size(x1,2),1);
        end
        
        theta1s    = [theta1s;theta1'];
        exit_infos = [exit_infos;exit_info];        

        vcov       = var_cov(theta2);
        se         = full(sqrt(diag(vcov)));
        std_errors = [std_errors;se'];        

        mvalolds2  = [mvalolds2,mvalolds];
        mvalold0   = mvalolds(:,1);
        mvalold00  = mvalolds(:,2);

        mvalold=mvalold00;
        g=gradobj(theta2);
        
        mvalold=mvalold0;
        g2=gradient_vec('gmmobj',theta2);
        
        mvalold=mvalold0;
        H = blp_hessian_mat('gmmobj',theta2,[]);
        eig_H=eig(reshape(H,size(theta2,1),size(theta2,1)));

        mvalold=mvalold00;
        options = optimset(...
                'MaxFunevals',1,...
                'GradObj',     'on',...
                'HessUpdate',  'bfgs',...
                'LargeScale',  'off',...
                'TolFun',      mytolfun,...
                'TolX',        mytolx,...
                'Display','off');

        [theta2b,fvalb,exit_infob,tmpb,g3,H2]=fminunc('gmmobj2',theta2,options);
        eig_H2=eig(reshape(H2,size(theta2,1),size(theta2,1)));

        %address irregular gradients and hessians
        if (min(g3)==-999999) && (max(g3)==-999999) && (std(g3)==0)           
            g      = -999999*ones(size(theta2));
            g2     = -999999*ones(size(theta2));
            g3     = -999999*ones(size(theta2));
            
            H      = -999999*ones(size(theta2,1),size(theta2,1));
            H2     = -999999*ones(size(theta2,1),size(theta2,1));
            
            eig_H  = -999999*ones(size(theta2));
            eig_H2 = -999999*ones(size(theta2));
        end    
                
        fprintf('\nObj. function:\t');
        printm(fval);        
        
        fprintf('\ntheta1:\t');
        printm(theta1');

        fprintf('\ntheta2\t');
        printm(theta2');
                
        fprintf('\ngradient-analytical\n');
        fprintf('%18.4f\n',g);

        fprintf('\ngradient-numerical I\n');
        fprintf('%18.4f\n',g2);
 
        fprintf('\ngradient-numerical II\n');
        fprintf('%18.4f\n',g3);

        fprintf('\nhessian eigenvalues-numerical I\n');
        fprintf('%18.4f\n',eig_H);                       

        fprintf('\nhessian eigenvalues-numerical II\n');
        fprintf('%18.4f\n',eig_H2);                       

        hessians    =[hessians;H(:)'];
        hessians2   =[hessians2;H(:)'];
        gradients   =[gradients;g'];
        gradients2  =[gradients2;g2'];            
        gradients3  =[gradients3;g3'];            
        
        toc_tmp=toc;
        tocs=[tocs;toc];
        
    end %perturbations loop 
    
    %**********************************************************************
    % Save results
    %**********************************************************************
    cd(results_path)
        fprintf('/n');
        fprintf('Saving optimization results.../n');
        save (matfile, 'perturbs2', 'fvals', 'theta1s', 'theta2s','exit_infos',...
             'hessians','hessians2','gradients', 'gradients2' ,'gradients3',...
             'deltas' ,'gmmresids' ,'mvalolds2','std_errors','counts2',...
             'fvals_track','tocs');
    cd(code_path)
    diary off

end

% ADDED CODE
theta1
theta2
se


%% 2. Setup
% clear all
% close all
% close hidden
% warning off all
% clc

%**************************************************************************
%Define paths for input and output folders
%**************************************************************************
code_path    =pwd;
optim_results_path          =[code_path,'/Optimization results/'];
logs_path                   =[code_path,'/Optimization logs/'];
add_path                    =[code_path,'/Optimization routines/'];
mkt_power_results_path      =[code_path,'/Market power results/'];
merger_results_path         =[code_path,'/Merger results/'];
addpath(add_path);

%**************************************************************************
%Define globals
%**************************************************************************
global ns x1 x2 vfull dfull theta1 cdid cdindex IV1 own nbrn alphai mvalold oldt2 s_jt

%**************************************************************************
%Load data and define various parameters, including ownership dummies
%**************************************************************************
load BLP_data cdid cdindex share outshr price firmid id const hpwt air mpd space mpg trend product model_id own_dummies
load BLP_data_str model_name

%% 2.a)
% Define Matrices
y = log([price]);
X = [const,(hpwt),air,(mpg),space];

% Coefficients
coefficients = inv(X'*X)*X'*y

% Robust Std. Errors
residuals = y - X*coefficients;
covmatrix = inv(X'*X)*(X'*diag(residuals.^2)*X)*inv(X'*X); 
robust_se = sqrt(diag(covmatrix))

coefficients
robust_se

%% 2.b)
s_jt=share;

% demand instruments
sum_other=[];
sum_rival=[];
X=[const,(hpwt),air,(mpg),space];
for i=1:size(id,1)
    other_ind=(firmid==firmid(i)  & cdid==cdid(i) & id~=id(i));
    rival_ind=(firmid~=firmid(i)  & cdid==cdid(i));
    sum_other(i,:)=sum(X(other_ind==1,:));
    sum_rival(i,:)=sum(X(rival_ind==1,:));
end
IV1=[X,sum_other,sum_rival];

%Load N(0,I) drwas
load v

%ownership structure matrix
own=own_dummies; clear owndummies

%variables in demand without random coefficients
x1=[price,X]; clear X

%variables in demand with random coefficients
x2=x1(:,1:size(x1,2)-1);

%#of indivduals and markets
ns =  size(v,2)/5;
nmkt = 20;

%#of brands per market
nbrn=zeros(nmkt,1);
nbrn(1)=cdindex(1);
for i=2:max(nmkt)
nbrn(i)=sum(cdid==i);
end

%demographics and N(0,I) unobservables
demogr=zeros(size(v));
vfull = v(cdid,:);
dfull = demogr(cdid,:);

%**************************************************************************
%Logit IV regression
%**************************************************************************
theta2w=     [3.612 0 0 0 0;
              4.628 0 0 0 0;
              1.818 0 0 0 0;
              1.050 0 0 0 0;
              2.056 0 0 0 0];
[theti, thetj, theta2]=find(theta2w);

invA1=inv(IV1'*IV1);
y = log(s_jt) - log(outshr);
mid = x1'*IV1*invA1*IV1';
t = inv(mid*x1)*mid*y;
oldt2 = zeros(size(theta2));
mvalold=x1*t;
mvalold_logit = mvalold;

mkt_power_results=[];

%**************************************************************************
%Loop over the various optimization routines
%**************************************************************************
for optrout=1:1

    %Optimization routine 6 (GA-JBES) did not produce reasonable results
    %in the optimization stage
    if optrout~=6
        
        cd(optim_results_path)

        if optrout<=9
            matfile=['blp_0',num2str(optrout),'_data_optim'];
            mkt_power_file=[mkt_power_results_path,'blp_mkt_power_results_0',num2str(optrout),'.txt'];            
            merger_file   =[merger_results_path,'blp_merger_results_0',num2str(optrout),'.txt'];            
        else
            matfile=['blp_',num2str(optrout),'_data_optim'];
            mkt_power_file=[mkt_power_results_path,'blp_mkt_power_results_',num2str(optrout),'.txt'];            
            merger_file   =[merger_results_path,'blp_merger_results_',num2str(optrout),'.txt'];                        
        end

        load (matfile, 'theta1s', 'theta2s','deltas','fvals','perturbs2');
        cd(code_path);

%        remove comments to perform the analysis only for the "best"
%        set of results
%        [min_fval,min_fval_ind]=min(fvals);
%        theta1s   = theta1s(min_fval_ind,:);
%        theta2s   = theta2s(min_fval_ind,:);
%        deltas    = deltas(:,min_fval_ind);
%        fvals     = fvals(min_fval_ind,:);
%        perturbs2 = perturbs2(min_fval_ind,:);

        mkt_power_results=[]; merger_results=[];

        for jj_optrout=1:1:size(fvals,1)

            theta1=theta1s(jj_optrout,:)';
            theta2=theta2s(jj_optrout,:)';
            delta=deltas(:,jj_optrout)';
            fval=fvals(jj_optrout,:);
            perturb=perturbs2(jj_optrout,:);

            mvalold=exp(mvalold_logit);
            oldt2 = zeros(size(theta2));
            deltajt=meanval(theta2);

            theta2w(:,1)=theta2;

            sijt=ind_sh(exp(deltajt),exp(mufunc(x2,theta2w)));
            sijt_pre=sijt;
            sjt_pre=(1/ns)*sum(sijt')';
            deltajt_pre=deltajt;

            vfull1=vfull(:,1:ns);
            alpha_i=[];
            for i=1:size(vfull1,1)
                alpha_i(i,:)=vfull1(i,:).*(kron(theta2(1),ones(1,ns)))+...
                    (kron(theta1(1),ones(1,ns)));
            end

            alphai=alpha_i;
            deriv_all=zeros(max(nbrn),max(nbrn),nmkt);
            elast_all=zeros(max(nbrn),max(nbrn),nmkt);

            for i=1:nmkt

                ind=cdid==i;
                pjt=price(ind==1,1);
                sjt=s_jt(ind==1,1);
                alpha_i=alphai(ind==1,:);
                s_ijt=sijt(ind==1,:);

                elast=zeros(size(pjt,1),size(pjt,1));
                deriv=zeros(size(pjt,1),size(pjt,1));

                for j=1:size(pjt,1)

                    deriv(j,j)=(1/ns)*sum(alpha_i(j,:).*s_ijt(j,:).*(ones(1,ns)-s_ijt(j,:)));
                    elast(j,j)=(pjt(j)./sjt(j))*(1/ns)*sum(alpha_i(j,:).*s_ijt(j,:).*(ones(1,ns)-s_ijt(j,:)));

                    for k=1:size(pjt,1)

                        if k~=j
                            elast(j,k)=-(pjt(k)./sjt(j))*(1/ns)*sum(alpha_i(j,:).*s_ijt(j,:).*(s_ijt(k,:)));
                            deriv(j,k)=-(1/ns)*sum(alpha_i(j,:).*s_ijt(j,:).*(s_ijt(k,:)));

                        end

                    end

                end

                elast_all(1:size(elast,1),1:size(elast,2),i)=elast;
                deriv_all(1:size(deriv,1),1:size(deriv,2),i)=deriv;

            end

            %store own and cross price elasticities
            elast_own=[];
            elast_cross=[];
            for i=1:nmkt
                temp=diag(elast_all(1:nbrn(i),1:nbrn(i),i));
                elast_own=[elast_own;temp];
                elast_cross=[elast_cross;elast_all(1:nbrn(i),:,i)];
            end

            %Consumer surplus calculations pre-merger
            exp_V=ind_eg(exp(deltajt),exp(mufunc(x2,theta2w)));
            tmp=[];
            CV_pre=[];
            for i=1:nmkt
                alphai_tmp=-alphai(cdid==i,:);
                alphai_tmp=alphai_tmp(1,:);
                tmp(i,:)=log(sum(exp_V(cdid==i,:))+1)./alphai_tmp;
                CV_pre(i,:)=tmp(i,:);
            end

            %Market_power calculation pre_merger
            own_dummy_pre=own;
            price_pre=price;

            mm=[];
            for i=1:max(cdid)
                p=price_pre(cdid==i,:);
                s=sjt_pre(cdid==i,:);
                nn=nbrn(i);
                om=deriv_all(1:nn,1:nn,i).*(own_dummy_pre(cdid==i,:)*own_dummy_pre(cdid==i,:)');
                m=-inv(om')*s;
                mm=[mm;m];
            end

            margin_pre=mm;
            mc=price_pre-margin_pre;
            margin_pct_pre=(margin_pre)./price_pre;
            profit_pre=margin_pre.*sjt_pre;

            optrout_aux=repmat(optrout,size(price,1),1);
            startval_aux=repmat(perturb,size(price,1),1);
            fval_aux=repmat(fval,size(price,1),1);
            market=cdid;

            mkt_power_results=[mkt_power_results;[optrout_aux,startval_aux,fval_aux,market,model_id,price_pre,sjt_pre,s_jt,elast_own,mc margin_pre,margin_pct_pre,elast_cross,mean(alphai')',std(alphai')']];

            %merger ownership matrix
            tmp=own(:,16)+own(:,19);
            own_dummy_post=[own(:,1:15),tmp,own(:,17:18),own(:,20:26)];

            mm=[];
            for i=1:max(cdid)
                p=price_pre(cdid==i,:);
                s=sjt_pre(cdid==i,:);
                nn=nbrn(i);
                om=deriv_all(1:nn,1:nn,i).*(own_dummy_post(cdid==i,:)*own_dummy_post(cdid==i,:)');
                m=-inv(om')*s;
                mm=[mm;m];
            end
            price_approx=mc+mm;
            price_post=price_approx;
            margin_post=mm;
            margin_pct_post=(margin_post)./price_post;

            %individual market shares post-merger
            deltajt_post=deltajt_pre-price_pre*theta1(1)+price_post*theta1(1);
            x2_post=x2;
            x2_post(:,1)=price_post;

            %calculate implied market shares
            theta2w = zeros(5,5);
            theta2w(:,1)=theta2;

            %update component of mu that corresponds to price
            [n k] = size(x2_post);
            j = size(theta2w,2)-1;
            mu = zeros(n,ns);
            for i = 1:ns
                v_i = vfull(:,i:ns:k*ns);
                d_i = dfull(:,i:ns:j*ns);
                mu(:,i) = (x2_post.*v_i*theta2w(:,1));
            end
            mu_post=mu;
            expmu=exp(mu);
            expmval=exp(deltajt_post);

            sijt_post=ind_sh(expmval,expmu);
            sjt_post=(1/ns)*sum(sijt_post')';

            %consumer surplus post-merger
            exp_V=ind_eg(expmval,expmu);
            tmp=[];
            CV_post=[];
            for i=1:nmkt
                alphai_tmp=-alphai(cdid==i,:);
                alphai_tmp=alphai_tmp(1,:);
                tmp(i,:)=log(sum(exp_V(cdid==i,:))+1)./alphai_tmp;
                CV_post(i,:)=tmp(i,:);
            end

            %profit post-merger
            profit_post=margin_post.*sjt_post;

            mean_CV=mean((CV_post-CV_pre)')';

            mean_CV_aux=[];
            for i=1:size(nbrn,1)
                tmp=nbrn(i);
                mean_CV_aux    =[mean_CV_aux;repmat(mean_CV(i,1),tmp,1)];
            end

            fprintf('optim routine : %2i\t',optrout);
            fprintf('start value   : %3i\t',perturb);
            fprintf('median elast  : %12.4f\n',median(diag(elast)));

            merger_results=[merger_results;[optrout_aux,startval_aux,market,model_id,price_pre,price_post,sjt_pre,sjt_post,mc,profit_pre,profit_post,mean_CV_aux]];
        end

        cd(mkt_power_results_path);     
        save(mkt_power_file,'mkt_power_results','-ASCII');
        
        cd(merger_results_path);     
        save(merger_file,'merger_results','-ASCII');
        
        cd(code_path);
                
    end    
end

theta1
theta2
se