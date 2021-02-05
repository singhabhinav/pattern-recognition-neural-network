path(path, '~\Matlab\NetLab\');
PATH = ['~\Demand\'];

load demand_data.txt  
data = demand_data(:,4:5);  % The first column of data is load
date = demand_data(:,1);  
xc = data';

num_data = length(data);  	
WIN = 30;                % Length of moving window for training
AHEAD = 7;	           % Steps ahead  to be predicted
LEVEL=4;                 % Number of wavelet decomposition
time_base=50;            % Parameter used in a trous filtering

% Normalize all data to [-1 +1] :
Me =  mean(xc,2);
mxc(1,:) = xc(1,:) - Me(1);
mxc(2,:) = xc(2,:) - Me(2);

Max = max(mxc(1,:)); 
Min = min(mxc(1,:));  
rc = 2*(mxc(1,:)-Min)/(Max-Min) - 1;
% Predict the load (demand value) series only

clear data xc

OPTION =menu('Simulation options','1. data preparation','2. training','3. testing','4. close');

if OPTION==4
    close all
end

if OPTION ~=1 

    name = [PATH '\demand_wvdata.mat']; 
    num_train = 1000;  
    num_test  = 400;   
    eval(['load ' name ]); 
    clear rc xc
end

if OPTION==2   % define parameters required by the MATTLAB


    alpha = 0.01;
    % coefficient of weight-decay
    func =  'linear';
    % output unit activation
    % alternatives: "linear","logistic", "softmax"

    prior = alpha;

    % Set up vector of options for optimiser, required by NETLAB 	 
    options = zeros(1,18);
    options(1) = 1;          % This provides display of error values.
    options(14) = 1000;      % Number of training cycles. 

end

if OPTION ==3   
    tsts = input('Testing on training set (y/n)? ', 's');
    if strcmp(tsts, 'y')
        start_tst = 1;
        end_tst = num_train;
        testtype='tr';
    else
        start_tst = num_train + 1;
        end_tst =  num_train + num_test;
        testtype='tst';
    end 

end


% *********************************************************
% *********************************************************

if OPTION == 1	

    % data pre-processing by shift-invariant wavelet transform	 
    % (a trous filtering by appling auto-correlation shell decomposition)


    [SER, DATE] = atfilter(rc, LEVEL, num_data, time_base, date);

    col = 0;
    while col < LEVEL+2
        col = col + 1;
        tmp =  SER(:,col)';
        eval(['WVSER' num2str(col-1) '=tmp;' ])             
    end 

    name = [PATH 'demand_wvdata.mat'];           
    eval(['save ' name  ' WVSER* DATE ']); 

    subplot(6,2,1)
    plot(WVSER0)    
    subplot(6,2,3)
    plot(WVSER1)
    subplot(6,2,5)
    plot(WVSER2)    
    subplot(6,2,7)
    plot(WVSER3)    
    subplot(6,2,9)
    plot(WVSER4)    
    subplot(6,2,11)
    plot(WVSER5)

end

% *********************************************************
% %%%%%%%   Training MLPs on different scales  %%%%%%%%%%%%


if OPTION == 2 

    % Using wavelet processed data, we train a number of
    % MLPs on different scales (including a raw price) 
    % In this scheme, different days ahead are predicted
    % by separate modules 

    % We use one perceptron or MLP to intergrate the results from different levels


    TX = WVSER0(1:num_train);	
    % Target series to be forecast
    [Ni,Np] = size(TX);
    nout = 1;         % number of outputs  in all MLPs
    nin= WIN;         % number of inputs
    nhidden=round((nin+nout)/2);  % number of hidden units
    % This is an emprical choice

    for lev =0: LEVEL+ 1    

        fprintf('Training for resolution level  %d\n',lev);  
        ser = eval(['WVSER' num2str(lev) ]);  
        trx= ser(1: num_train);   clear ser

        for D=1:AHEAD     
            if lev~=LEVEL+1	
                fprintf('MLP training for step = %d\n',D);  
                k = 0;
                while k< Np - (nin+D-1)
                    k = k + 1;
                    px(k,:) = trx(k:k+nin-1);
                    pt(k,:) = trx(k+nin+D-1);
                end 
                nn = mlp(nin,nhidden,nout,func,alpha);	            
                nn = netopt(nn, options, px,pt, 'scg'); 
                eval(['net' num2str(lev) '.step' num2str(D) '=nn;'])           
                eval(['p' num2str(lev) '.step' num2str(D) '=pt;']) 	    
                clear px pt tmpt xin z o            
            else	% regression for residual    	   
                fprintf('Regression for step = %d\n',D);  
                k = 0;
                while k< Np - (nin+D-1)
                    k = k + 1;
                    px(k,:) = trx(k:k+nin-1);
                    pt(k,:) = trx(k+nin+D-1);
                end
                coef = pinv(px'*px)*(px')*pt;   	         	    
                eval(['p' num2str(lev) '.step' num2str(D) '= pt;']); 
                eval(['net' num2str(lev) '.step' num2str(D) '=coef;'])   
                clear coef px pt	 
            end	     		
        end	       % complete training for all steps ahead
        % in one level of wavelet coefficients
    end 	       
    % complete training for all levels of wavelet coefficients
    % This is the first stage of prediction

    %
    nin=WIN; 

    lev = -1;  P = [];      
    while lev< Np - (WIN+AHEAD-1)
        k = k + 1;
        tmpP(k,d) = tmp(k,1);
    end 
    clear tmp
end 	      
P = [P beta*tmpP];
clear tmpP tmp
end    
clear p*
k=0;
while k< Np -(nin+D-1)
    k= k + 1;      
    xin = tstx(k:k+nin-1);   
    tmpt(k,1)= tstx(k+nin+D-1);                   
    nd = size(xin, 1);                       
    z = tanh(xin*mlpnet.w1 + ones(nd, 1)*mlpnet.b1);
    tmpo(k,:) = z*mlpnet.w2 + ones(nd, 1)*mlpnet.b2;  	
end              
eval(['p' num2str(lev) '.step' num2str(D) '= tmpo;'])
eval(['py' num2str(lev) '.step' num2str(D) '= tmpt;']);
clear tmpo tmpt tmpa   xin z      	

else		% regression for residual    

    nin= WIN;
    coef =eval(['net' num2str(lev) '.step' num2str(D)]);
    k = 0;
    while k< length(tstx) - (nin+D-1)
        k = k + 1;
        px(k,:) = tstx(k:k+nin-1); 
    end  
    py = tstx(nin+D:Np);
    pred = px*coef;    
    eval(['p' num2str(lev) '.step' num2str(D) '= pred;']);
    eval(['py' num2str(lev) '.step' num2str(D) '= py;']);
    clear  coef px py pred 				
end

end		%complete prediction for all steps in one level

end 		%complete prediction for all levels


% Monitor prediction for each level
% "one-step-ahead" prediction      
for lev=0:LEVEL+1      
    fprintf('Predict for level = %d\n',lev);	
    for d=1:AHEAD
        fprintf('Predict for step = %d\n',d);	
        pred = eval(['p' num2str(lev) '.step' num2str(d) ]);  
        % Predicted series
        dest = eval(['py' num2str(lev) '.step' num2str(d) ]); 
        % Target series corresponds to pred  
        plot(pred(101:200),'r')
        hold on; pause(1);
        plot(dest(101:200),'g')
        hold off; pause(1);  %pause;    
    end
end	


P = [];   wvp= 0;   
lev = -1;    
while lev< Np - (nin+AHEAD-1)
    k = k + 1;
    tmpP(k,d) = tmp(k);
end 
clear tmp
end 		
P = [P beta*tmpP];

if lev ~=0
    wvp = wvp + beta*tmpP;
else
    mlpp = tmpP;	
end 	
clear tmpP tmp
end    

% For the plain target, wvp is the results from the 
% direct summation of wavelet coefficients predictions 
% mlpp:  prediction from a single, independent MLP

% Prediction from 2nd stage perceptron
k = 0;        
while k < Np-(WIN+D-1) 
    k = k +1;
    target(k,:)=X(WIN+k:WIN+AHEAD+k-1);
    Tdate(k,1)=Date(WIN+k);

    xin = P(k,:);                 
    hyp1(k, :) = xin*W;     % W is perceptron's weight; 
    % prediction from perceptron
    ndata = size(xin, 1);
    z = tanh(xin*NET.w1 + ones(ndata, 1)*NET.b1);
    hyp2(k,:) = z*NET.w2 + ones(ndata, 1)*NET.b2; 		
end                      



% Change back to original data range:   
% 	Pwv is from the scheme by direct summation
% 	Phy is from hybrid scheme prediction
% 	Ptar is target series
% 	Pmlp is from a single MLP prediction

Ptar = 0.5*(Max-Min)*target + 0.5*(Max+Min)+ Me(1);
Phy1 = 0.5*(Max-Min)*hyp1 + 0.5*(Max+Min)+ Me(1);
Phy2 = 0.5*(Max-Min)*hyp2 + 0.5*(Max+Min)+ Me(1);
Pwv = 0.5*(Max-Min)*wvp + 0.5*(Max+Min)+ Me(1);
Pmlp = 0.5*(Max-Min)*mlpp + 0.5*(Max+Min)+ Me(1);

% Performance calculations:
for j=1:length(Tdate)
    tmper1 = Ptar(j,:) - Pmlp(j,:);
    tmper2 = Ptar(j,:) - Phy1(j,:); 
    tmper3 = Ptar(j,:) - Phy2(j,:); 
    tmper4 = Ptar(j,:) - Pwv(j,:); 
    for k=1:AHEAD
        mlpape(j,k) = abs(tmper1(k))/Ptar(j,k);
        hy1ape(j,k) = abs(tmper2(k))/Ptar(j,k);
        hy2ape(j,k) = abs(tmper3(k))/Ptar(j,k);
        wvape(j,k) = abs(tmper4(k))/Ptar(j,k);         
    end
end

for step=1:AHEAD
    mean_mlpape(step) = mean(mlpape(step,:));
    var_mlpape(step) = var(mlpape(step,:));
    prc_mlpape(step) = prctile(mlpape(step,:),90);

    mean_hy1ape(step) = mean(hy1ape(step,:));
    var_hy1ape(step) = var(hy1ape(step,:));
    prc_hy1ape(step) = prctile(hy1ape(step,:),90);

    mean_hy2ape(step) = mean(hy2ape(step,:));
    var_hy2ape(step) = var(hy2ape(step,:));
    prc_hy2ape(step) = prctile(hy2ape(step,:),90);

    mean_wvape(step) = mean(wvape(step,:));
    var_wvape(step) = var(wvape(step,:));
    prc_wvape(step) = prctile(wvape(step,:),90);

    nmse_mlper(step) = nmse(Ptar(:,step), Pmlp(:,step) );
    nmse_hy1er(step) = nmse(Ptar(:,step), Phy1(:,step) );
    nmse_hy2er(step) = nmse(Ptar(:,step), Phy2(:,step) );
    nmse_wver(step) = nmse(Ptar(:,step), Pwv(:,step) );
end

figure 
subplot(321)
plot(Pmlp(:,1));		hold on
plot(Ptar(:,1),'r -.');
title('One step ahead prediction (MLP)')
subplot(322)
plot(Pmlp(:,AHEAD));		hold on
plot(Ptar(:,AHEAD),'r -.');
title('Seven steps ahead prediction (MLP) ')
subplot(323)
plot(Pwv(:,1));		hold on
plot(Ptar(:,1),'r -.');
title('One step ahead prediction (a trous)')
subplot(324)
plot(Pwv(:,AHEAD));		hold on
plot(Ptar(:,AHEAD),'r -.');
title('Seven steps ahead prediction (a trous) ')
subplot(325)
plot(Phy1(:,1));		hold on
plot(Ptar(:,1),'r -.');
title('One step ahead prediction (hybrid)')
subplot(326)
plot(Phy1(:,AHEAD));		hold on
plot(Ptar(:,AHEAD),'r -.');
title('Seven steps ahead prediction (hybrid)')

figure
subplot(321)
plot(mlpape(:,1));		hold on
title('APE for one step prediction (MLP)')
subplot(322)
plot(mlpape(:,AHEAD));		hold on
title('APE for seven step prediction (MLP)')

subplot(323)
plot(wvape(:,1));		hold on
title('APE for one step prediction (a trous)') 
subplot(324)
plot(wvape(:,AHEAD));		hold on
title('APE for seven step prediction (a trous)')

subplot(325)
plot(hy2ape(:,1));		hold on
title('APE for one step prediction (hybrid)') 
subplot(326)
plot(hy2ape(:,AHEAD));		hold on
title('APE for seven step prediction (hybird)')

if strcmp(tsts, 'y')
    name3 = [PATH 'demand_err_tr.mat'];            
else
    name3 = [PATH 'demand_err_tst.mat'];        
end    
eval(['save ' name3 ' mlpape wvape hy1ape hy2ape mean* var* prc* nmse* ']) 

end
