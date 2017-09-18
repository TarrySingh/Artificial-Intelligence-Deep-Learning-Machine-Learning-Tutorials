function Ycs = cosmoother_pca(Y, params)
%
% Ycs = cosmoother_pca(Y, params)
%
% Performs leave-neuron-out prediction for PCA.
%
% INPUTS:
%
% Y           - test data (# neurons x # data points)
% params      - PCA parameters fit to training data
%
% OUTPUTS:
%
% Ycs         - leave-neuron-out prediction (# neurons x # data points)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu
  
  L  = params.L;
  d  = params.d;
  
  [yDim, xDim] = size(L);

  Ycs = zeros(size(Y));  

  for i = 1:yDim  
    % Indices 1:yDim with i removed
    mi = [1:(i-1) (i+1):yDim];

    Xmi = inv(L(mi,:)' * L(mi,:)) * L(mi,:)' *... 
          bsxfun(@minus, Y(mi,:), d(mi));
    
    Ycs(i,:) = L(i,:) * Xmi + d(i);  
  end
