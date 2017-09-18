function [ssm, like] = ssm_em (data, varargin)
% smm_em - fit an LGSSM to data using the EM algorithm
%
%       [Params, like] = SSM_EM (X, ...) fits the SSM to the data in X
%       [nObservables x nTimes x nSequences] using the EM algorithm.
%       It returns the new model as well the log-likelihood after each
%       iteration, like.
%
%       OPTIONS:
%
%       'nIter' -       Maximum number of iterations of the EM algorithm.
%	'tol' -		Convergence tolerance: when change in
%			likelihood per point per step drops below
%			this threshold, iteration is stopped.
%       'latentdim' -   number of latent dimensions to use

init  = [];
nIter = 100;
tol = 1e-5;
latentdim = 1;

optlistassign(who, varargin{:});

% useful inline function
cellsum = @(C)(sum(cat(3, C{:}), 3));

% discover dimensions
[DD, nTime, nSeq] = size(data);         % Output dim; # timepoints; # sequences
KK = latentdim;			        % Latent dim

if (isempty(init))
  y0 = zeros(KK,1);
  Q0 = eye(KK);
  A  = randn(KK,KK);
  Q  = eye(KK);
  C  = randn(DD,KK);
  R  = eye(DD);
else
  A = init.dynamics;
  C = init.output;
  Q = init.innovations;
  R = init.noise;
  y0 = init.initstate; 
  Q0 = init.initvar;
end


like = zeros(1, nIter);			% allocate likelihood list


% precalculate some useful stuff
Exx  = reshape(data, DD, [])*reshape(data, DD, [])';

%% run EM
iIter = 0;
while (iIter < nIter)			% avoid 'for' to allow nIter=Inf
  iIter = iIter + 1;

  Eyy = zeros(KK,KK);
  Ey2y2 = zeros(KK,KK);
  Ey1y1 = zeros(KK,KK);
  Ey2y1 = zeros(KK,KK);
  Exy = zeros(DD,KK);
  
  for iSeq = 1:nSeq
    [yhat, Vhat, Vjoint, ll]  = ssm_kalman(data(:,:,iSeq), y0, Q0, A, Q, C, R, 'smooth');
  
    if (any(cellfun(@det, Vhat) < 0))
      warn('ssm_em: non PSD variance');
    end

    Eyy_trial = yhat*yhat' + cellsum(Vhat);
    
    Eyy   = Eyy + Eyy_trial;
    Ey2y2 = Ey2y2 + Eyy_trial - yhat(:,1)*yhat(:,1)' - Vhat{1};
    Ey1y1 = Ey1y1 + Eyy_trial - yhat(:,end)*yhat(:,end)' - Vhat{end};

    Ey2y1 = Ey2y1 + yhat(:,2:end)*yhat(:,1:end-1)' + cellsum(Vjoint);
    Exy   = Exy + data(:,:,iSeq)*yhat';
    
    like(iIter) = like(iIter) + sum(ll);
  end

  C = Exy / Eyy;
  R = (Exx - Exy*C')/(nSeq*nTime);  
  R = diag(diag(R));                    % keep only diagonal variance

  A = Ey2y1/Ey1y1;
  Q = (Ey2y2 - Ey2y1*A')/(nSeq*(nTime-1));
  Q = (Q+Q')/2;                         % symmetrise to avoid accumulated numerical issues

  % don't update these
  %   y0 = 
  %   Q0 = 

  fprintf('\rSSM em iteration: %d \t\t likelihood = %g', iIter, like(iIter));

  if tol > 0 & iIter > 1
    if abs(diff(like(iIter-1:iIter))) < tol*diff(like([1,iIter]))
      like = like(1:iIter);             % truncate
      break;
    end
  end

end  
fprintf('\n');

ssm.dynamics = A;
ssm.output = C;
ssm.innovations = Q;
ssm.noise = R;
ssm.initstate = y0; 
ssm.initvar = Q0;
  
