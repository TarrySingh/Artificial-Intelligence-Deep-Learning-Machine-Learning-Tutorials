function res = learnGPparams(seq, params, varargin)
% Updates parameters of GP state model given neural trajectories.
%
% INPUTS:
%
% seq         - data structure containing neural trajectories
% params      - current GP state model parameters, which gives starting point
%               for gradient optimization
%
% OUTPUT:
%
% res         - updated GP state model parameters
%
% OPTIONAL ARGUMENTS:
%
% MAXITERS    - maximum number of line searches (if >0), maximum number 
%               of function evaluations (if <0), for minimize.m (default:-8)
% verbose     - logical that specifies whether to display status messages
%               (default: false)
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu

  MAXITERS  = -8; % for minimize.m
  verbose   = false;
  assignopts(who, varargin);

  switch params.covType
    case 'rbf'
      % If there's more than one type of parameter, put them in the
      % second row of oldParams.
      oldParams = params.gamma;
      fname     = 'grad_betgam';
    case 'tri'
      oldParams = params.a;
      fname     = 'grad_trislope';
    case 'logexp'
      oldParams = params.a;
      fname     = 'grad_logexpslope';
  end
  if params.notes.learnGPNoise
    oldParams = [oldParams; params.eps];
    fname     = [fname '_noise'];
  end

  xDim    = size(oldParams, 2);
  precomp = makePrecomp(seq, xDim);
  
  % Loop once for each state dimension (each GP)
  for i = 1:xDim
    const = [];
    switch params.covType
      % No constants for 'rbf' or 'tri'
      case 'logexp'
        const.gamma = params.gamma;
    end
    if ~params.notes.learnGPNoise  
      const.eps = params.eps(i);     
    end

    initp = log(oldParams(:,i));

    % This does the heavy lifting
    [res_p, res_f, res_iters] =...
    minimize(initp, fname, MAXITERS, precomp(i), const);
    
    switch params.covType
      case 'rbf'
        res.gamma(i) = exp(res_p(1));
      case 'tri'
        res.a(i)     = exp(res_p(1));
      case 'logexp'
        res.a(i)     = exp(res_p(1));
    end    
    if params.notes.learnGPNoise  
      res.eps(i) = exp(res_p(2));
    end
      
    if verbose
      fprintf('\nConverged p; xDim:%d, p:%s', i, mat2str(res_p, 3));
    end
  end
