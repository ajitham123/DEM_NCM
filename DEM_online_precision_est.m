%%  Noise Covariance Estimation for colored noise using 
%   Dynamic Expectation Maximization algorithm.

% Author: Ajith Anil Meera, Department of Artificial Intelligence,
% Radbound University, Nijmegen

% Date: 31st August 2023
% Code accompanying the paper "Adaptive Noise Covariance Estimation
% under Colored Noise using Dynamic Expectation Maximization.
% link: https://arxiv.org/pdf/2308.07797.pdf

%%%%%%%%%%%%%
%%

clear all;

% y =  [Av; Rv; Ap]
% u = [Ea; Er; Ap] stimulus

model.C = [.8 0 0.2; 0 1 0; .3 0 .7]; 
% brain.C = model.C;
brain.C = [.8 0 0.2; 0 1 0; 0 .3 .7]; 


ny = size(model.C,1);
nu = size(model.C,2);

model.T = 32;
model.dt = .1;
model.nt = 1 + model.T/model.dt;
model.t = 0:model.dt:model.T;

brain.prior.h = [10; 10; 10].*ones(ny,1);
% brain.prior.h = [9 7 5 3]'.*ones(ny,1);
brain.prior.Ph = diag([exp(-4) exp(-4) exp(-4)])*diag(ones(1,ny));
brain.prior.u = zeros(nu,model.nt);
brain.prior.Pu = exp(0)*diag(ones(nu,1));

% brain.prior.h = [12; 12];
% brain.prior.Ph = diag([exp(16) exp(16)]);

brain.h = zeros(ny,model.nt);
brain.h(:,1) = brain.prior.h;
brain.u = zeros(nu,model.nt);

% Initialise optimal conditional precisions
brain.Pi.h = zeros(ny,ny,model.nt);
brain.Pi.h(:,:,1) = brain.prior.Ph;
brain.Pi.u = zeros(nu,nu,model.nt);
brain.Pi.u(:,:,1) = brain.prior.Pu;
brain.std.u = zeros(nu,model.nt);
brain.std.u(:,1) =  sqrt(diag(inv(brain.prior.Pu))); %1./sqrt(diag(brain.prior.Pu));

brain.F = zeros(1,model.nt);
k_h = 4;     k_u = 4;

model.y = zeros(ny,model.nt); model.z = zeros(ny,model.nt);


% Define the causes of the generative process
model.u = [1; .5; .75].*ones(nu,1).*(exp(-.25*(model.t-8).^2) + ...
                    1* exp(-.25*(model.t-20).^2));

% Define a time varying noise precision diagonals
model.Pz = exp(smoothdata(([9; 7; 8].*[ones(ny,floor(model.nt/2)) ...
                                           zeros(ny,model.nt - floor(model.nt/2))]...
                     + [-1; 7; 8].*[zeros(ny,floor(model.nt/2)) ...
                            ones(ny,model.nt - floor(model.nt/2))])',...
                                           'gaussian',floor(model.nt/5))');

% model.Pz = exp(8)*ones(ny,);
     
% Compute free energy
[F, dFdh, dFdhh, dFdu, dFduu] = free_energy(ny, nu, brain.C, brain.prior);

% fixed seed white noise
rng(12);
white_noise = randn(ny,model.nt);

h = sym('h',[ny 1]);             y = sym('y',[ny 1]);
u = sym('u',[nu 1]);             prior_u = sym('prior_u',[nu 1]);
Pi_u = sym('Pi_u',[nu nu]);      Pi_h = sym('Pi_h',[ny ny]);

for i = 2:model.nt
    
    % Generative process
    model.z(:,i) = sqrtm(inv(diag(model.Pz(:,i))))*white_noise(:,i);
    model.y(:,i) = model.C*model.u(:,i) + model.z(:,i);
    
    % update the conditional precision for hyperparameters
    brain.Pi.h(:,:,i) = brain.prior.Ph;
    
    for k = 1:ny
        eval(sprintf('%s=brain.h(k,i-1);',char(h(k))));
        eval(sprintf('%s=model.y(k,i);',char(y(k))));
        for kk = 1:ny
            eval(sprintf('%s=brain.Pi.h(k,kk,i);',char(Pi_h(k,kk))));
        end
    end
    
    % update the conditional precision for inputs
    brain.Pi.u(:,:,i) = -eval(dFduu);   %%%%%%%%% only when trace terms are not used for Free energy expression
    brain.std.u(:,i)  = sqrt(diag(pinv(brain.Pi.u(:,:,i)))); %1./sqrt(diag(brain.Pi.u(:,:,i))); 
    
    for k = 1:nu
        eval(sprintf('%s=brain.u(k,i-1);',char(u(k))));
        eval(sprintf('%s=brain.prior.u(k,i);',char(prior_u(k))));
        for kk = 1:nu
            eval(sprintf('%s=brain.Pi.u(k,kk,i);',char(Pi_u(k,kk))));
        end
    end
                
    brain.F(1,i) = eval(F);    
    
    % update hyperparameters
    dh = ((expm(k_h*eval(dFdhh)*model.dt)-eye(ny))*pinv(eval(dFdhh))*eval(dFdh));
    brain.h(:,i) = brain.h(:,i-1) + dh;
    
    % update input estimates
    du = ((expm(k_u*eval(dFduu)*model.dt)-eye(nu))*pinv(eval(dFduu))*eval(dFdu));
    brain.u(:,i) = brain.u(:,i-1) + du;
    
    
    
%     if i == 50
%         figure(13); clf
%         temp_F = []; temp_h1 = [];
%         for j = -1:.1:15
%             eval(sprintf('%s=j;',char(h(1))));
%             temp_F = [temp_F eval(F)];
%             temp_h1 =  [temp_h1 eval(h(1))];          
%         end
%         figure(13);  plot(temp_h1,temp_F,'r*'); hold on; ylim([-2 9]);
%         xlabel('hyperparameter h1'); ylabel('Free energy')
%     end
end

figure(12); clf;  plot(0:model.dt:model.T, (brain.h')); hold on; 
plot(0:model.dt:model.T,log((model.Pz')),'k-.'); hold on;
legend('\lambda^{Av} est','\lambda^{Rv} est','\lambda^{Ap} est',...
       '\lambda real','location','northeast')
xlabel('Time (s)'); ylabel('noise hyperparameter')

figure(13); clf; 
for i =1:nu
    subplot(nu,1,i)
    mean_2 = brain.u(i,:)';
    std_2 = brain.std.u(i,:)';
    fill([model.t fliplr(model.t)],...
        [full(mean_2 + std_2)' fliplr(full(mean_2 - std_2)')],...
        [1 1 1]*.8,'EdgeColor',[1 1 1]*.8)
    hold on;  
    plot(model.t,mean_2,'b'); hold on; plot(model.t,model.u(i,:)','r'); hold on; 
    legend('\sigma^{}','est','real')
end
subplot(nu,1,1); ylabel('Ea'); subplot(nu,1,2); ylabel('Er'); subplot(nu,1,3); ylabel('Ap');
xlabel('time(s)')
suptitle('State/input estimation');

% plot precision on estimated states
figure(14); clf; plot(model.t,squeeze(brain.Pi.u(1,1,:))); hold on;
plot(model.t,squeeze(brain.Pi.u(2,2,:))); hold on;
plot(model.t,squeeze(brain.Pi.u(3,3,:))); hold on;
set(gca, 'YScale', 'log')
legend('\Pi^{Ea}','\Pi^{Er}','\Pi^{Ap}')
plot(model.t,1./(brain.std.u.^2),'k'); hold on;


% Plot cross correlation between states
cross_cov = zeros(1,model.nt);
for i = 1:model.nt
   cov = pinv(brain.Pi.u(:,:,i));
   cross_cov(i) = cov(2,3);
end
figure(15); clf; plot(cross_cov')

% figure(14); clf; plot(brain.F); ylim([0 25])

% sys.F = 0;  sys.H = model.C;
% est = benchmark(y,sys);


function [F, dFdh, dFdhh, dFdu, dFduu] = free_energy(ny,nu,C,prior)

h = sym('h',[ny 1]);
y = sym('y',[ny 1]);
u = sym('u',[nu 1]);
prior_u = sym('prior_u',[nu 1]);

Pi_u = sym('Pi_u',[nu nu]);
Pi_h = sym('Pi_h',[ny ny]);

Pi = diag(exp(h));


%%%%%%% Add the trace terms to F later for accurate estimation
F = -.5*transpose(y - C*u)*Pi*(y - C*u) + .5*log(det(Pi)) + ...
    -.5*transpose(h - prior.h)*prior.Ph*(h - prior.h)  + ...
    -.5*transpose(u - prior_u)*prior.Pu*(u - prior_u)  + ...
    -.5*( log(det(Pi_u)) + log(det(Pi_h)) ) ; %+ .5*log(det(prior.Ph)) + .5*log(det(prior.Pu));
%     dFdh = zeros(ny,1);
%     dFdhh = zeros(ny,ny);
for i = 1:ny
    dFdh(i,1) = diff(F,h(i));
    for j = 1:ny
        dFdhh(i,j) = diff(dFdh(i,1),h(j)); % + constants
    end
end

for i = 1:nu
    dFdu(i,1) = diff(F,u(i));
    for j = 1:nu
        dFduu(i,j) = diff(dFdu(i,1),u(j)); % + constants
    end
end
end



function est = benchmark(y,sys)


% CMM PARAMETERS
%--------------------------------------------------
% initial state estimate
CMMpar.xp = param.xp;
CMMpar.Pp = eye(2);

% initial estimates of Q and R
CMMpar.Q = eye(2);
CMMpar.R = eye(2);

% initial time instant for matrices estimation
CMMpar.erq = floor(N/2);



%  GMBM PARAMETERS
%--------------------------------------------------
% initial state estimates
GMBMpar.xp = param.xp;                                                                                                                                                        
GMBMpar.Pp = eye(2);

% qunatised matrices for Q and R
Qquant = zeros(2,2,4);
Qquant(:,:,1) = sys.Q;                                                                                                                                                       
Qquant(:,:,2) = [2 0;0 5];                                                                                                                                                       
Qquant(:,:,3) = [2 -1;-1 2];                                                                                                                                                     
Qquant(:,:,4) = [2 1;1 3];                                                                                                                                                                                                                                                                                                                                
GMBMpar.Qquant = Qquant;

Rquant = zeros(2,2,2);
Rquant(:,:,1) = [3 0;0 1.5];                                                                                                                                                       
Rquant(:,:,2) = sys.R;                                                                                                                                                        
GMBMpar.Rquant = Rquant; 



%  VBM PARAMETERS
%--------------------------------------------------
VBMpar.rho = @(N)1-min(50./(N),1e-1); % behaviour of precision factor

% parameters of inverse Gamma distribution
VBMpar.alfa = [0;0]; 
VBMpar.beta = [1;1];

VBMpar.itCnt = 2; % # of filtering iterations

% initial state estimate
VBMpar.xp = param.xp;
VBMpar.Pp = zeros(2);
param.rho = @(N)1-min(50./(N),1e-1);




est.CMM = CMM(sys,y,CMMpar);
est.GMBM = GMBM(sys,y,GMBMpar);
est.VBM = VBM(sys,z,VBMpar);



end


function est=CMM(sys,z,param)
  %CMM (sys,z,param) Covariance Matching Method
  %
  % DCM - Section 4.1
  %
  % based on:
  % K. A. Myers, B. D. Tapley, "Adaptive sequential estimation with unknown
  % noise statistics", IEEE Transactions on Automatic Control,
  % vol. 21, no. 8, pp. 520-523, 1976
  %
  % estimates Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP, PARAM.PP describe initial estimate of the state
  % PARAM.Q, PARAM.R initial estimates of Q and R
  % PARAM.EQR initial time instant for matrices estimation
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  
  xf = zeros(nx,N);
  Pf = zeros(nx,nx,N);
  innov = zeros(nz,N);
  R = zeros(nz,nz,N);
  Gam = zeros(nz,nz,N);
  de = zeros(nx,nx,N);
  Q = zeros(nx,nx,N);
  q = zeros(nx,N);
  r = zeros(nz,N);
  K = zeros(nx,nz,N);
  
  xp = param.xp;
  erq = param.erq;
  Pp = param.Pp;
  Q(:,:,1) = param.Q;
  R(:,:,1) = param.R;
  Q(:,:,erq) = param.Q;
  R(:,:,erq) = param.R;
  
  for i = 1:N
    if i <= erq
      % filter algorithm (no Q,R estimation)
      
      % filtering step
      K(:,:,i) = Pp*sys.H'/(sys.H*Pp*sys.H'+R(:,:,1));
      innov(:,i) = z(:,i)-sys.H*xp;
      xf(:,i) = xp+K(:,:,i)*innov(:,i);
      Pf(:,:,i) = (eye(nx)-K(:,:,i)*sys.H)*Pp;
      
      % auxiliary variables for R estimation
      r(:,i) = innov(:,i);
      Gam(:,:,i) = sys.H*Pp*sys.H';
      
      % auxiliary variables for Q estimation
      if i > 1
        q(:,i) = xf(:,i)-sys.F*xf(:,i-1);
        de(:,:,i) = sys.F*Pf(:,:,i-1)*sys.F'-Pf(:,:,i);
      end
      
      % prediction step
      xp = sys.F*xf(:,i);
      Pp = sys.F*Pf(:,:,i)*sys.F'+Q(:,:,1);
    else
      % CMM Algorithm
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - ESTIMATE THE MEASUREMENT NOISE CM R
      r(:,i) = z(:,i)-sys.H*xp;
      Gam(:,:,i) = sys.H*Pp*sys.H';
      R(:,:,i) = cov(r(:,i-erq:i)') - mean(Gam(:,:,i-erq:i),3);
      
      %%%%%%%%%%%%%%%%%%%%%%%%%% STEP II - ESTIMATE THE FILTERING STATE ESTIMATE
      K(:,:,i) = Pp*sys.H'/(sys.H*Pp*sys.H'+R(:,:,i));
      innov(:,i) = r(:,i);
      xf(:,i) = xp+K(:,:,i)*innov(:,i);
      Pf(:,:,i) = (eye(nx)-K(:,:,i)*sys.H)*Pp;
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP III - ESTIMATE THE STATE NOISE CM Q
      q(:,i) = xf(:,i)-sys.F*xf(:,i-1);
      de(:,:,i) = sys.F*Pf(:,:,i-1)*sys.F'-Pf(:,:,i);
      Q(:,:,i) = cov(q(:,i-erq:i)') - mean(de(:,:,i-erq:i),3);
      
      %%%%%%%%%%%%%%%%%%%%%%%%% STEP IV - ESTIMATE THE PREDICTION STATE ESTIMATE
      xp = sys.F*xf(:,i);
      Pp = sys.F*Pf(:,:,i)*sys.F'+Q(:,:,i);
    end
  end
%   est.Q = squeeze(Q(:,:,N));
%   est.R = squeeze(R(:,:,N));
    est.Q = Q; 
    est.R = R;
end

function [est] = GMBM(sys,z,param)
  %GMM (sys,z,param) Gaussian Mixture Bayesian Method
  %
  % GMM - Section 4.2
  %
  % based on:
  % D. G. Lainiotis, "Optimal adaptive estimation: Structure and parameters
  % adaptation", IEEE Transactions on Automatic Control, vol. 16, no. 2,
  % pp. 160-170, 1971.
  %
  % estimates Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP describes initial estimate of the state
  % PARAM.PP describes initial estimate of the state variance
  % PARAM.Qquant quantised matrices for Q
  % PARAM.Rquant quantised matrices for R
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  
  nq = size(param.Qquant,3); % number of basis matrices for Q
  nr = size(param.Rquant,3); % number of basis matrices for R
  
  est.Q = zeros(nx,nx,N);
  est.R = zeros(nz,nz,N);
  
  weight = ones(nq*nr,N);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - DESIGN SET OF nq*nr LINEAR FILTERS
  xp = cell(nq*nr);
  Pxp = cell(nq*nr);
  xf = cell(nq*nr);
  Pxf = cell(nq*nr);
  for i = 1:nq
    for j = 1:nr
      xp{i+(j-1)*nq} = param.xp;
      Pxp{i+(j-1)*nq} = param.Pp;
      xf{i+(j-1)*nq} = zeros(size(param.xp));
      Pxf{i+(j-1)*nq} =zeros(size(param.Pp));
    end
  end
  
  %%%%%%%%%% STEP II - RUN EACH FILTER AND EVALUATE THE A POSTERIORI PROBABILITY
  for k = 1:N
    for i = 1:nq
      for j = 1:nr
        % evaluation of the likelihood function
        weight(i+(j-1)*nq,k) = 1/(sqrt((2*pi)^nz*det(param.Rquant(:,:,j))))...
          *exp(-0.5*(z(:,k)-sys.H*xp{i+(j-1)*nq})'...
          /(param.Rquant(:,:,j))*(z(:,k)-sys.H*xp{i+(j-1)*nq}));
        % filtering
        K = Pxp{i+(j-1)*nq}*sys.H'...
          /(sys.H*Pxp{i+(j-1)*nq}*sys.H'+param.Rquant(:,:,j));
        xf{i+(j-1)*nq} = xp{i+(j-1)*nq}+K*(z(:,k)-sys.H*xp{i+(j-1)*nq}); 
        Pxf{i+(j-1)*nq} = (eye(nx)-K*sys.H)*Pxp{i+(j-1)*nq};
        % prediction
        xp{i+(j-1)*nq} = sys.F*xf{i+(j-1)*nq}; 
        Pxp{i+(j-1)*nq} = sys.F*Pxf{i+(j-1)*nq}*sys.F'+param.Qquant(:,:,i);
      end
    end
    if k > 1
        weight(:,k) =weight(:,k).*weight(:,k-1);
    end
    weight(:,k) = weight(:,k)/sum(weight(:,k));
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP III - ESTIMATE Q and R
    Qq = zeros(nx);
    Rr = zeros(nz);
    for i = 1:nq
        for j = 1:nr
            Qq = Qq+weight(i+(j-1)*nq,k)*param.Qquant(:,:,i);
            Rr = Rr+weight(i+(j-1)*nq,k)*param.Rquant(:,:,j);
        end
    end
    est.Q(:,:,k) = Qq;
    est.R(:,:,k) = Rr;
  end
end


function [est] = VBM(sys,z,param)
  %ICM (sys,z,param) Variational Bayes Method
  %
  % ICM - Section 4.4
  %
  % based on:
  % S. Sarkka, A Nummenmaa, "Recursive noise adaptive Kalman filtering by
  % variational bayesian approximations", IEEE Transactions on Automatic
  % Control, vol. 54, no. 3, pp. 596-600, 2009.
  %
  % estimates diagonal elements of R
  % SYS.F, SYS.H and sys.Q are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP, PARAM.PP describes initial estimate of the state and its variance
  % PARAM.ALFA, PARAM.BETA are initial estimates of inverse Gamma distribution
  % PARAM.ITCNT determines number of filtering iterations (2-3 are recommended)
  % PARAM.RHO(N) governs behavior of precision factor (recomm.: 1-alfa) alfa->0
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  
  est.R = zeros(nz.nz,N);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I MODEL and ESTIMATE INITIALIZATION
  xp = param.xp;
  Pp = param.Pp;
  Rest = zeros(nz,N);
  alfap = param.alfa;
  betap = param.beta;
  %%%%%%%%%%%%%%%%%%%%%%%%% STEP II + III RECURSIVE ESTIMATION OF STATE AND CM R 
  for i = 1:N
    alfaf = ones(nz,1)/2+alfap; % parameter adjustment
    betaf = betap; 
    innov = (z(:,i)-sys.H*xp); % innovation
    
    %filtration/iteration
    for j = 1:param.itCnt
      Rest(:,i) = betaf./alfaf; % estimation of R
      K = Pp*sys.H'/(sys.H*Pp*sys.H'+diag(Rest(:,i))); % filter gain
      xf = xp+K*innov; % state estimate filtering update
      Pf = Pp-K*sys.H*Pp; % filtering state covariance update
      betaf = betap+(z(:,i)-sys.H*xf).^2/2+diag(sys.H*Pf*sys.H')/2;% par. change
    end
    
    %prediction
    xp = sys.F*xf; % state estimate time update
    Pp = sys.F*Pf*sys.F'+sys.Q; % variance time update
    alfap = param.rho(N)*alfaf; % parameter alfa time update
    betap = param.rho(N)*betaf; % parameter beta time update
    
    est.R(:,:,i) = diag(Rest(:,i));
  end
  
end




