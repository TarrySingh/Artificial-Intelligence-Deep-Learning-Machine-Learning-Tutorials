function [yhat, Vhat, Vjoint, like] = ssm_kalman(xx, y0, Q0, A, Q, C, R, pass)
% SSM_KALMAN - kalman-smoother estimates of SSM state posterior
%
% [Y,V,Vj,L] = SSM_KALMAN(X,Y0,Q0,A,Q,C,R) peforms the Kalman
% smoothing recursions on the (DxT) data matrix X, for the
% LGSSM defined by the following parameters
%  y0 Kx1 - initial latent state
%  Q0 KxK - initial variance
%  A  KxK - latent dynamics matrix
%  Q  KxK - innovariations covariance matrix
%  C  DxK - output loading matrix
%  R  DxD - output noise matrix
% The function returns:
%  Y  KxT - posterior mean estimates
%  V  1xT cell array of KxK matrices - posterior variances on y_t
%  Vj 1xT-1 cell array of KxK matrices - posterior covariances between y_{t+1}, y_t
%  L  1xT - conditional log-likelihoods log(p(x_t|x_{1:t-1}))
%
% [Y,V,[],L] = SSM_KALMAN(..., 'filt') performs Kalman filtering.
% The joint covariances (Vj) are not computed and an empty cell
% array is returned.

% check for ssm structure instead of individual arguments
if (nargin < 7) && isstruct(y0)
  if (nargin < 3)
    pass = 'smooth';
  else
    pass = Q0;
  end
  ssm = y0;
  y0 = ssm.initstate;
  Q0 = ssm.initvar;
  A = ssm.dynamics;
  Q = ssm.innovations;
  C = ssm.output;
  R = ssm.noise;
elseif nargin < 8
  pass = 'smooth';
end

% check dimensions

[dd,kk] = size(C);
[tt] = size(xx, 2);

if any([size(y0) ~= [kk,1], ...
        size(Q0) ~= [kk,kk], ...
        size(A) ~=  [kk,kk], ...
        size(Q) ~=  [kk,kk], ...
        size(R) ~=  [dd,dd]])
  error ('inconsistent parameter dimensions');
end



%%%% allocate arrays

yfilt = zeros(kk,tt);                  % filtering estimate: \hat(y)_t^t
Vfilt = cell(1,tt);                    % filtering variance: \hat(V)_t^t
yhat  = zeros(kk,tt);                  % smoothing estimate: \hat(y)_t^T
Vhat  = cell(1,tt);                    % smoothing variance: \hat(V)_t^T
K = cell(1, tt);                       % Kalman gain
J = cell(1, tt);                       % smoothing gain
like = zeros(1, tt);                   % conditional log-likelihood: p(x_t|x_{1:t-1})

Ik = eye(kk);

invR = diag(1./diag(R));
invRC = invR*C;
CinvRC = C'*invR*C;

%%%% forward pass

Vpred = Q0;
ypred = y0;

for t = 1:tt
  xprederr = xx(:,t) - C*ypred;


  Vfilt{t} = inv(inv(Vpred) + CinvRC);
  %% symmetrise to avoid numerical drift
  Vfilt{t} = (Vfilt{t} + Vfilt{t}')/2;

  %% Vxpred = C*Vpred*C'+R;
  invVxpred = invR - invRC*Vfilt{t}*invRC';
  
  %% like(t) = -0.5*logdet(2*pi*(Vxpred)) - 0.5*xprederr'/Vxpred*xprederr;
  like(t) = 0.5*logdet(2*pi*(invVxpred)) - 0.5*xprederr'*invVxpred*xprederr;

  %%  K{t} = Vpred*C/Vxpred;
  K{t} = Vfilt{t}*invRC';

  yfilt(:,t) = ypred + K{t}*xprederr;
%   Vfilt{t} = Vpred - K{t}*C*Vpred;
  
  ypred = A*yfilt(:,t);
  Vpred = A*Vfilt{t}*A' + Q;
end


%%%% backward pass

if (strncmp(lower(pass), 'filt', 4) || strncmp(lower(pass), 'forw', 4))
  % skip if filtering/forward pass only
  yhat = yfilt;
  Vhat = Vfilt;
  Vjoint = {};
else
  yhat(:,tt) = yfilt(:,tt);
  Vhat{tt}   = Vfilt{tt};

  for t = tt-1:-1:1
    J{t} = (Vfilt{t}*A')/(A*Vfilt{t}*A' + Q);
    yhat(:,t) = yfilt(:,t) + J{t}*(yhat(:,t+1) - A*yfilt(:,t));
    Vhat{t}   = Vfilt{t} + J{t}*(Vhat{t+1} - A*Vfilt{t}*A' - Q)* J{t}';
  end

  Vjoint{tt-1} = (Ik - K{tt}*C)*A*Vfilt{tt-1};
  for t = tt-2:-1:1
    Vjoint{t} = Vfilt{t+1}*J{t}' + J{t+1}*(Vjoint{t+1} - A*Vfilt{t+1})*J{t}';
  end
end
