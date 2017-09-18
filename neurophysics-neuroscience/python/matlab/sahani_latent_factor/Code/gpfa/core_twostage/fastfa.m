function [estParams, LL] = fastfa(X, zDim, varargin)
%
% [estParams, LL] = fastfa(X, zDim, ...) 
%
% Factor analysis and probabilistic PCA.
%
%   xDim: data dimensionality
%   zDim: latent dimensionality
%   N:    number of data points
%
% INPUTS: 
%
% X    - data matrix (xDim x N)
% zDim - number of factors
%
% OUTPUTS:
%
% estParams.L  - factor loadings (xDim x zDim)
% estParams.Ph - diagonal of uniqueness matrix (xDim x 1)
% estParams.d  - data mean (xDim x 1)
% LL           - log likelihood at each EM iteration
%
% OPTIONAL ARGUMENTS:
%
% typ        - 'fa' (default) or 'ppca'
% tol        - stopping criterion for EM (default: 1e-8)
% cyc        - maximum number of EM iterations (default: 1e8)
% minVarFrac - fraction of overall data variance for each observed dimension
%              to set as the private variance floor.  This is used to combat 
%              Heywood cases, where ML parameter learning returns one or more 
%              zero private variances. (default: 0.01)
%              (See Martin & McDonald, Psychometrika, Dec 1975.)
% verbose    - logical that specifies whether to display status messages
%              (default: false)
%
% Code adapted from ffa.m by Zoubin Ghahramani.
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  typ        = 'fa';
  tol        = 1e-8; 
  cyc        = 1e8;
  minVarFrac = 0.01;
  verbose    = false;
  assignopts(who, varargin);

  randn('state', 0);
  [xDim, N] = size(X);
    
  % Initialization of parameters
  cX    = cov(X', 1);
  if rank(cX) == xDim
    scale = exp(2*sum(log(diag(chol(cX))))/xDim);
  else
    % cX may not be full rank because N < xDim
    fprintf('WARNING in fastfa.m: Data matrix is not full rank.\n');
    r     = rank(cX);
    e     = sort(eig(cX), 'descend');
    scale = geomean(e(1:r));
  end
  L     = randn(xDim,zDim)*sqrt(scale/zDim);
  Ph    = diag(cX);
  d     = mean(X, 2);

  varFloor = minVarFrac * diag(cX);  

  I     = eye(zDim);
  const = -xDim/2*log(2*pi);
  LLi   = 0; 
  LL    = [];
  
  for i = 1:cyc
    % =======
    % E-step
    % =======
    iPh  = diag(1./Ph);
    iPhL = iPh * L;  
    MM   = iPh - iPhL / (I + L' * iPhL) * iPhL';
    beta = L' * MM; % zDim x xDim
    
    cX_beta = cX * beta'; % xDim x zDim
    EZZ     = I - beta * L + beta * cX_beta;
        
    % Compute log likelihood
    LLold = LLi;    
    ldM   = sum(log(diag(chol(MM))));
    LLi   = N*const + N*ldM - 0.5*N*sum(sum(MM .* cX)); 
    if verbose
      fprintf('EM iteration %5i lik %8.1f \r', i, LLi);
    end
    LL = [LL LLi];    
    
    % =======
    % M-step
    % =======
    L  = cX_beta / EZZ;
    Ph = diag(cX) - sum(cX_beta .* L, 2);
    
    if isequal(typ, 'ppca')
      Ph = mean(Ph) * ones(xDim, 1);
    end
    if isequal(typ, 'fa')
      % Set minimum private variance
      Ph = max(varFloor, Ph);
    end
       
    if i<=2
      LLbase = LLi;
    elseif (LLi < LLold)
      disp('VIOLATION');
    elseif ((LLi-LLbase) < (1+tol)*(LLold-LLbase))
      break;
    end
  end

  if verbose
    fprintf('\n');
  end

  if any(Ph == varFloor)
    fprintf('Warning: Private variance floor used for one or more observed dimensions in FA.\n');
  end

  estParams.L  = L;
  estParams.Ph = Ph;
  estParams.d  = d;
