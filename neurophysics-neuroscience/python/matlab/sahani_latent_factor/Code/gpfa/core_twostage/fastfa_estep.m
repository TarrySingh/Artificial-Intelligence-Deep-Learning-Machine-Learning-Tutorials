function [Z, LL] = fastfa_estep(X, params)
%
% [Z, LL] = fastfa_estep(X, params)
%
% Compute the low-dimensional points and data likelihoods using a
% previously learned FA or PPCA model.
%
%   xDim: data dimensionality
%   zDim: latent dimensionality
%   N:    number of data points
%
% INPUTS:
%
% X        - data matrix (xDim x N)
% params   - learned FA or PPCA parameters (structure with fields L, Ph, d)
%
% OUTPUTS:
%
% Z.mean   - posterior mean (zDim x N)
% Z.cov    - posterior covariance (zDim x zDim), which is the same for all data
% LL       - log-likelihood of data
% 
% Note: the choice of FA vs. PPCA does not need to be specified because  
% the choice is reflected in params.Ph.
%
% Code adapted from ffa.m by Zoubin Ghaharamani.
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  [xDim, N] = size(X);
  zDim      = size(params.L, 2);
    
  L     = params.L;
  Ph    = params.Ph;
  d     = params.d;
  
  Xc   = bsxfun(@minus, X, d);
  XcXc = Xc * Xc';

  I=eye(zDim);
    
  const=-xDim/2*log(2*pi);
    
  iPh  = diag(1./Ph);
  iPhL = iPh * L;    
  MM   = iPh - iPhL / (I + L' * iPhL) * iPhL';
  beta = L' * MM; % zDim x xDim
    
  Z.mean = beta * Xc; % zDim x N
  Z.cov  = I - beta * L; % zDim x zDim; same for all observations

  LL = N*const + 0.5*N*logdet(MM) - 0.5 * sum(sum(MM .* XcXc));
