function seq = cosmoother_gpfa_viaOrth(seq, params, mList)
%
% seq = cosmoother_gpfa_viaOrth(seq, params, mList)
%
% Performs leave-neuron-out prediction for GPFA.
%
% INPUTS:
%
% seq         - test data structure
% params      - GPFA model parameters fit to training data
% mList       - number of top orthonormal latent coordinates to use for 
%               prediction (e.g., 1:5)
%
% OUTPUTS:
%
% seq         - test data structure with new fields ycsOrthXX, where XX are
%               elements of mList.  seq(n).ycsOrthXX has the same dimensions
%               as seq(n).y.
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu

  [yDim, xDim] = size(params.C);

  for n = 1:length(seq)
    for m = mList
      fn          = sprintf('ycsOrth%02d', m);
      seq(n).(fn) = nan(yDim, seq(n).T);
    end
  end

  for i = 1:yDim    
    % Indices 1:yDim with i removed
    mi = [1:(i-1) (i+1):yDim];
    
    for n = 1:length(seq)
      seqCs(n).T = seq(n).T;
      seqCs(n).y = seq(n).y(mi,:);
    end
    paramsCs   = params;
    paramsCs.C = params.C(mi,:);
    paramsCs.d = params.d(mi);
    paramsCs.R = params.R(mi,mi);
    
    seqCs = exactInferenceWithLL(seqCs, paramsCs, 'getLL', false);
    
    % Note: it is critical to use params.C here and not paramsCs.C 
    [Xorth, Corth]   = orthogonalize([seqCs.xsm], params.C);
    seqCs            = segmentByTrial(seqCs, Xorth, 'xorth');
    
    for n = 1:length(seq)
      for m = mList
        fn               = sprintf('ycsOrth%02d', m);
        seq(n).(fn)(i,:) = Corth(i,1:m) * seqCs(n).xorth(1:m,:) + params.d(i);
      end
    end
    fprintf('Cross-validation complete for %3d of %d neurons\r', i, yDim);
  end
  fprintf('\n');
