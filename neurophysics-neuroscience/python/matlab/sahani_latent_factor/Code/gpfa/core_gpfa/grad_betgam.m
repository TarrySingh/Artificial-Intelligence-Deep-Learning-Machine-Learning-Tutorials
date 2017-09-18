function [f, df] = grad_betgam(p, precomp, const)
%
% [f, df] = grad_betgam(p, precomp, const)  
%
% Gradient computation for GP timescale optimization.
% This function is called by minimize.m.
%
% INPUTS:
%
% p           - variable with respect to which optimization is performed,
%               where p = log(1 / timescale ^2)
% precomp     - structure containing precomputations
%
% OUTPUTS:
%
% f           - value of objective function E[log P({x},{y})] at p
% df          - gradient at p    
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu
  
  Tall = precomp.Tall;
  Tmax = max(Tall);
    
  temp         = (1-const.eps) * exp(-exp(p(1)) / 2 * precomp.difSq); % Tmax x Tmax
  Kmax         = temp + const.eps * eye(Tmax);
  dKdgamma_max = -0.5 * temp .* precomp.difSq;
  
  dEdgamma = 0;
  f        = 0;
  for j = 1:length(precomp.Tu)
    T     = precomp.Tu(j).T;
    Thalf = ceil(T/2);
            
    [Kinv, logdet_K] = invToeplitz(Kmax(1:T, 1:T));
    
    KinvM     = Kinv(1:Thalf,:) * dKdgamma_max(1:T,1:T); % Thalf x T
    KinvMKinv = (KinvM * Kinv)';                         % Thalf x T
    
    dg_KinvM  = diag(KinvM);
    tr_KinvM  = 2 * sum(dg_KinvM) - rem(T, 2) * dg_KinvM(end);
    
    mkr = ceil(0.5 * T^2);
    
    dEdgamma = dEdgamma - 0.5 * precomp.Tu(j).numTrials * tr_KinvM...
    + 0.5 * precomp.Tu(j).PautoSUM(1:mkr) * KinvMKinv(1:mkr)'... 
    + 0.5 * precomp.Tu(j).PautoSUM(end:-1:mkr+1) * KinvMKinv(1:(T^2-mkr))';
        
    f = f - 0.5 * precomp.Tu(j).numTrials * logdet_K...
    - 0.5 * precomp.Tu(j).PautoSUM(:)' * Kinv(:);
  end
  
  f  = -f;
  % exp(p) is needed because we're computing gradients with
  % respect to log(gamma), rather than gamma
  df = -dEdgamma * exp(p(1));
