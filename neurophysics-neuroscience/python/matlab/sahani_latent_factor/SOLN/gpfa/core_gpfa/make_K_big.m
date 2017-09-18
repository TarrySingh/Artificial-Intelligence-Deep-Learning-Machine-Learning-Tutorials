function [K_big, K_big_inv, logdet_K_big] = make_K_big(params, T)
%
% [K_big, K_big_inv] = make_K_big(params, T)
%
% Constructs full GP covariance matrix across all state dimensions and timesteps.
%
% INPUTS:
%
% params       - GPFA model parameters
% T            - number of timesteps
%
% OUTPUTS:
%
% K_big        - GP covariance matrix with dimensions (xDim * T) x (xDim * T).
%                The (t1, t2) block is diagonal, has dimensions xDim x xDim, and 
%                represents the covariance between the state vectors at
%                timesteps t1 and t2.  K_big is sparse and striped.
% K_big_inv    - inverse of K_big
% logdet_K_big - log determinant of K_big
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu

  xDim = size(params.C, 2);

  idx = 0 : xDim : (xDim*(T-1));
    
  K_big        = zeros(xDim*T);
  K_big_inv    = zeros(xDim*T);
  Tdif         = repmat((1:T)', 1, T) - repmat(1:T, T, 1);
  logdet_K_big = 0;
  
  for i = 1:xDim
    switch params.covType
      case 'rbf'
        K = (1 - params.eps(i)) * ...
        exp(-params.gamma(i) / 2 * Tdif.^2) +...
        params.eps(i) * eye(T);
      case 'tri'
        K = max(1 - params.eps(i) - params.a(i) * abs(Tdif), 0) +...
        params.eps(i) * eye(T);
      case 'logexp'
        z = params.gamma *...
        (1 - params.eps(i) - params.a(i) * abs(Tdif));
        outUL = (z>36);
        outLL = (z<-19);
        inLim = ~outUL & ~outLL;
        
        hz        = nan(size(z));
        hz(outUL) = z(outUL);
        hz(outLL) = exp(z(outLL));
        hz(inLim) = log(1 + exp(z(inLim)));
        
        K = hz / params.gamma + params.eps(i) * eye(T);
    end
    K_big(idx+i, idx+i)                 = K;
    [K_big_inv(idx+i, idx+i), logdet_K] = invToeplitz(K);

    logdet_K_big = logdet_K_big + logdet_K;
  end
