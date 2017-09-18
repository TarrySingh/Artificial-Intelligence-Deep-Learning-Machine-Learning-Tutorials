function gpfaEngine(seqTrain, seqTest, fname, varargin)
%  
% gpfaEngine(seqTrain, seqTest, fname, ...) 
%
% Extract neural trajectories using GPFA.
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
% xDim          - state dimensionality (default: 3)
% binWidth      - spike bin width in msec (default: 20)
% startTau      - GP timescale initialization in msec (default: 100)
% startEps      - GP noise variance initialization (default: 1e-3)
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu

  xDim          = 3;
  binWidth      = 20; % in msec
  startTau      = 100; % in msec
  startEps      = 1e-3;
  extraOpts     = assignopts(who, varargin);
    
  % For compute efficiency, train on equal-length segments of trials
  seqTrainCut = cutTrials(seqTrain, extraOpts{:});
  if isempty(seqTrainCut)
    fprintf('WARNING: no segments extracted for training.  Defaulting to segLength=Inf.\n');
    seqTrainCut = cutTrials(seqTrain, 'segLength', Inf);
  end
  
  % ==================================
  % Initialize state model parameters
  % ==================================
  startParams.covType = 'rbf';
  % GP timescale
  % Assume binWidth is the time step size.
  startParams.gamma = (binWidth / startTau)^2 * ones(1, xDim);
  % GP noise variance
  startParams.eps   = startEps * ones(1, xDim);

  % ========================================
  % Initialize observation model parameters
  % ========================================
  fprintf('Initializing parameters using factor analysis...\n');
  
  yAll             = [seqTrainCut.y];
  [faParams, faLL] = fastfa(yAll, xDim, extraOpts{:});
  
  startParams.d = mean(yAll, 2);
  startParams.C = faParams.L;
  startParams.R = diag(faParams.Ph);

  % Define parameter constraints
  startParams.notes.learnKernelParams = true;
  startParams.notes.learnGPNoise      = false;
  startParams.notes.RforceDiagonal    = true;

  currentParams = startParams;

  % =====================
  % Fit model parameters
  % =====================
  fprintf('\nFitting GPFA model...\n');
  
  [estParams, seqTrainCut, LL, iterTime] =... 
    em(currentParams, seqTrainCut, extraOpts{:});
    
  % Extract neural trajectories for original, unsegmented trials
  % using learned parameters
  [seqTrain, LLorig] = exactInferenceWithLL(seqTrain, estParams);

  % ========================================
  % Leave-neuron-out prediction on test data
  % ========================================
  if ~isempty(seqTest) % check if there are any test trials
    if estParams.notes.RforceDiagonal
      seqTest = cosmoother_gpfa_viaOrth_fast(seqTest, estParams, 1:xDim);
    else
      seqTest = cosmoother_gpfa_viaOrth(seqTest, estParams, 1:xDim);
    end
  end
  
  % =============
  % Save results
  % =============
  vars  = who;
  fprintf('Saving %s...\n', fname);
  save(fname, vars{~ismember(vars, {'yAll'})});
