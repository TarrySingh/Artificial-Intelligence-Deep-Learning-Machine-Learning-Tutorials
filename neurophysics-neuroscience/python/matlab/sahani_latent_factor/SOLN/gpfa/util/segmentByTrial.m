function seq = segmentByTrial(seq, X, fn)
%
% seq = segmentByTrial(seq, X, fn)
%
% Segment and store data by trial.
%  
% INPUT:
%
% seq        - data structure that has field T, the number of timesteps
% X          - data to be segmented 
%                (any dimensionality x total number of timesteps)
% fn         - new field name of seq where segments of X are stored
%
% OUTPUT:
%
% seq       - data structure with new field 'fn'
%
% @ 2009 Byron Yu -- byronyu@stanford.edu
  
  if sum([seq.T]) ~= size(X, 2)
    fprintf('Error: size of X incorrect.\n');
  end
  
  ctr = 0;
  for n = 1:length(seq)
    T   = seq(n).T;
    idx = (ctr+1) : (ctr+T);
    seq(n).(fn) = X(:, idx);
    
    ctr = ctr + T;
  end
