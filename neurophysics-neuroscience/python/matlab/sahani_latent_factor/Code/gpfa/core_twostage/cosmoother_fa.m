function [Ycs, Vcs] = cosmoother_fa(Y, params)
%
% [Ycs, Vcs] = cosmoother_fa(Y, params)
%
% Performs leave-neuron-out prediction for FA or PPCA.
%
% INPUTS:
%
% Y           - test data (# neurons x # data points)
% params      - model parameters fit to training data using fastfa.m
%
% OUTPUTS: 
% 
% Ycs         - leave-neuron-out prediction mean (# neurons x # data points)
% Vcs         - leave-neuron-out prediction variance (# neurons x 1)
%
% Note: the choice of FA vs. PPCA does not need to be specified because
% the choice is reflected in params.Ph.
%
% @ 2009 Byron Yu -- byronyu@stanford.edu
  
  L  = params.L;
  Ph = params.Ph;
  d  = params.d;
  
  [yDim, xDim] = size(L);
  I = eye(xDim);

  Ycs = zeros(size(Y));  
  if nargout == 2
    % One variance for each observed dimension
    % Doesn't depend on observed data
    Vcs = zeros(yDim, 1);
  end

  for i = 1:yDim  
    % Indices 1:yDim with i removed
    mi = [1:(i-1) (i+1):yDim];
    
    Phinv  = 1./Ph(mi);                            % (yDim-1) x 1
    LRinv  = (L(mi,:) .* repmat(Phinv, 1, xDim))'; %     xDim x (yDim - 1)
    LRinvL = LRinv * L(mi,:);                      %     xDim x xDim
    
    term2 = L(i,:) * (I - LRinvL / (I + LRinvL)); % 1 x xDim
    
    dif      = bsxfun(@minus, Y(mi,:), d(mi));
    Ycs(i,:) = d(i) + term2 * LRinv * dif;

    if nargout == 2
      Vcs(i) = L(i,:)*L(i,:)' + Ph(i) - term2 * LRinvL * L(i,:)';
    end
  end
