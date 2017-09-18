function twoStageEngine(seqTrain, seqTest, fname, varargin)
%
% twoStageEngine(seqTrain, seqTest, fname, ...) 
%
% Extract neural trajectories using a two-stage method.
%
% INPUTS:
%
% seqTrain      - training data structure, whose nth entry (corresponding to
%                 the nth experimental trial) has fields
%                   trialId (1 x 1)   -- unique trial identifier
%                   y (# neurons x T) -- neural data
%                   T (1 x 1)         -- number of timesteps
% seqTest       - test data structure (same format as seqTrain)
% fname         - filename of where results are saved
%
% OPTIONAL ARGUMENTS:
%
% typ           - type of dimensionality reduction
%                 'fa' (default), 'ppca', 'pca'
% xDim          - state dimensionality (default: 3)
% binWidth      - spike bin width in msec (default: 20)
% kernSDList    - vector of Gaussian smoothing kernel widths to run
%                 Values are standard deviations in msec (default: 20:5:80)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu
  
  typ           = 'fa';
  xDim          = 3;
  binWidth      = 20; % in msec
  kernSDList    = 20:5:80; % in msec
  extraOpts     = assignopts(who, varargin);

  for k = 1:length(kernSDList)
    fprintf('Performing ''smooth and %s'' with kernSD=%d...\n',... 
    upper(typ), kernSDList(k));
    
    % ======================
    % Smooth data over time
    % ======================
    kern(k).kernSD   = kernSDList(k);
    
    % Training data
    kern(k).seqTrain = seqTrain;
    for n = 1:length(seqTrain)
      kern(k).seqTrain(n).y = smoother(seqTrain(n).y, kernSDList(k), binWidth);    
    end
    
    % Test data
    kern(k).seqTest = seqTest;
    for n = 1:length(seqTest)
      kern(k).seqTest(n).y = smoother(seqTest(n).y, kernSDList(k), binWidth);
    end

    YtrainRaw = [seqTrain.y];
    YtestRaw  = [seqTest.y];
    
    % ===============================
    % Apply dimensionality reduction
    % ===============================
    Y = [kern(k).seqTrain.y];
    
    if ismember(typ, {'fa', 'ppca'})
      [estParams, LL] = fastfa(Y, xDim, 'typ', typ, extraOpts{:});
      X               = fastfa_estep(Y, estParams);
      % To save disk space, don't save posterior covariance, which is identical
      % for each data point and can be computed from the learned parameters.
      kern(k).seqTrain  = segmentByTrial(kern(k).seqTrain, X.mean, 'xpost');
      kern(k).estParams = estParams;
      kern(k).LL        = LL;
          
    elseif isequal(typ, 'pca')
      [pcDirs, pcScores] = princomp(Y');
      
      kern(k).seqTrain = segmentByTrial(kern(k).seqTrain,... 
      pcScores(:,1:xDim)', 'xpost');  
           
      kern(k).estParams.L = pcDirs(:, 1:xDim);
      kern(k).estParams.d = mean(Y, 2);      
    end    
    
    
    % ========================================
    % Leave-neuron-out prediction on test data
    % ========================================
    if ~isempty(seqTest) % check if there are any test trials
      Y = [kern(k).seqTest.y];
      
      if ismember(typ, {'fa', 'ppca'})
        Ycs             = cosmoother_fa(Y, kern(k).estParams);
        kern(k).seqTest = segmentByTrial(kern(k).seqTest, Ycs, 'ycs');
        
      elseif isequal(typ, 'pca')
        Ycs             = cosmoother_pca(Y, kern(k).estParams);      
        kern(k).seqTest = segmentByTrial(kern(k).seqTest, Ycs, 'ycs');    
      end
    end
  end

  % =============
  % Save results
  % =============
  vars  = who;
  fprintf('Saving %s...\n', fname);
  save(fname, vars{~ismember(vars, {'X', 'Y', 'LL', 'estParams',... 
  'seqTrain', 'seqTest', 'pcDirs', 'pcScores', 'res'})});
