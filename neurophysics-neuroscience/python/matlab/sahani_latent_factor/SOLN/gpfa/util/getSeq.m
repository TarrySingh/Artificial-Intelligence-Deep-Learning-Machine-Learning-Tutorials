function seq = getSeq(dat, binWidth, varargin)
%
% seq = getSeq(dat, binWidth, ...)  
%
% Converts 0/1 spike trains into spike counts.
%
% INPUTS:
%
% dat         - structure whose nth entry (corresponding to the nth experimental
%               trial) has fields
%                 trialId -- unique trial identifier
%                 spikes  -- 0/1 matrix of the raw spiking activity across
%                            all neurons.  Each row corresponds to a neuron.
%                            Each column corresponds to a 1 msec timestep.
% binWidth    - spike bin width in msec
%
% OUTPUTS:
%
% seq         - data structure, whose nth entry (corresponding to
%               the nth experimental trial) has fields
%                 trialId      -- unique trial identifier
%                 T (1 x 1)    -- number of timesteps
%                 y (yDim x T) -- neural data
%
% OPTIONAL ARGUMENTS:
%
% useSqrt     - logical specifying whether or not to use square-root transform
%               on spike counts (default: true)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  useSqrt = true;
  assignopts(who, varargin);

  seq = [];
  for n = 1:length(dat)
    yDim = size(dat(n).spikes, 1);
    T    = floor(size(dat(n).spikes, 2) / binWidth);

    seq(n).trialId = dat(n).trialId;
    seq(n).T       = T;
    seq(n).y       = nan(yDim, T);
    
    for t = 1:T
      iStart = binWidth * (t-1) + 1;
      iEnd   = binWidth * t;
      
      seq(n).y(:,t) = sum(dat(n).spikes(:, iStart:iEnd), 2);
    end
    
    if useSqrt
      seq(n).y = sqrt(seq(n).y);
    end
  end
  
  % Remove trials that are shorter than one bin width
  if ~isempty(seq)
    trialsToKeep = ([seq.T] > 0);
    seq = seq(trialsToKeep);
  end
