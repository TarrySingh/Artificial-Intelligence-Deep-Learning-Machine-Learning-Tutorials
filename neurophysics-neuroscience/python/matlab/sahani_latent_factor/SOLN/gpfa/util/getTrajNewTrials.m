function seqNew = getTrajNewTrials(ws, dat, varargin)
%
% seqNew = getTrajNewTrials(ws, dat,...)
%
% Extract neural trajectories from a set of new trials using previously-fitted
% model parameters.
%
% INPUT:
%
% ws        - saved workspace variables that include the previously-fitted
%             model parameters 'estParams'
% dat       - data for new trials with fields
%               trialId -- unique trial identifier
%               spikes  -- 0/1 matrix of the raw spiking activity across
%                          all neurons.  Each row corresponds to a neuron.
%                          Each column corresponds to a 1 msec timestep.
%
% OUTPUTS:
%
% seqNew    - data structure containing orthonormalized neural trajectories
%             ('xorth') for the new trials
%
% OPTIONAL ARGUMENT:
%
% kernSD    - for two-stage methods, specify kernel smoothing width.  
%             By default, the function uses ws.kern(1).
%
% @ 2009 Byron Yu -- byronyu@stanford.edu
  
  kernSD = [];
  assignopts(who, varargin);

  if isempty(ws)
    fprintf('ERROR: Input argument is empty.\n');
    return
  end

  % Process data in the same way as in 'ws'    
  % Obtain binned spike counts
  seqNew  = getSeq(dat, ws.binWidth, ws.extraOpts{:});
  
  % Remove inactive units
  for n = 1:length(seqNew)
    seqNew(n).y = seqNew(n).y(ws.hasSpikesBool,:);
  end

  if isfield(ws, 'kernSDList')
    % Two-stage methods
    if isempty(kernSD)
      k = 1;
    else
      k = find(ws.kernSDList == kernSD);
      if isempty(k)
        fprintf('ERROR: Selected kernSD not found.\n');
        return
      end
    end
    
    % Kernel smoothing
    for n = 1:length(seqNew)
      seqNew(n).y = smoother(seqNew(n).y, ws.kernSDList(k), ws.binWidth);
    end
  end

  if ismember(ws.method, {'gpfa'})
    seqNew         = exactInferenceWithLL(seqNew, ws.estParams);
    C              = ws.estParams.C;
    X              = [seqNew.xsm];
    [Xorth, Corth] = orthogonalize(X, C);
    seqNew         = segmentByTrial(seqNew, Xorth, 'xorth');
    
  elseif ismember(ws.method, {'fa', 'ppca'})
    Y              = [seqNew.y];
    X              = fastfa_estep(Y, ws.kern(k).estParams);
    L              = ws.kern(k).estParams.L;
    [Xorth, Lorth] = orthogonalize(X.mean, L);
    seqNew         = segmentByTrial(seqNew, Xorth, 'xorth');
    
  elseif ismember(ws.method, {'pca'})
    Y              = [seqNew.y];
    estParams      = ws.kern(k).estParams;
    Xorth          = estParams.L' * bsxfun(@minus, Y, estParams.d);
    seqNew         = segmentByTrial(seqNew, Xorth, 'xorth');
  end
