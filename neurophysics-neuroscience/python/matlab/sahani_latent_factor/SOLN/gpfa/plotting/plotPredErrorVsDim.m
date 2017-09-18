function method = plotPredErrorVsDim(runIdx, kernSD, varargin)
%
% method = plotPredErrorVsDim(runIdx, kernSD,...)
%
% Plot prediction error versus state dimensionality.
%
% INPUTS:
%
% runIdx    - results files will be loaded from mat_results/runXXX, where
%             XXX is runIdx
% kernSD    - smoothing kernel standard deviation to use for two-stage methods
%
% OUTPUTS:
%
% method    - data structure containing prediction error values shown in plot
%
% OPTIONAL ARGUMENTS:
%
% plotOn    - logical that specifies whether or not to display plot
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  plotOn = true;
  assignopts(who, varargin);

  allMethods = {'pca', 'ppca', 'fa', 'gpfa'};
  
  runDir = sprintf('mat_results/run%03d', runIdx);
  if ~isdir(runDir)
    fprintf('ERROR: %s does not exist.  Exiting...\n', runDir);
    return
  else    
    D = dir([runDir '/*.mat']);
  end

  if isempty(D)
    fprintf('ERROR: No valid files.  Exiting...\n');
    method = [];
    return;
  end

  for i = 1:length(D)
    P = parseFilename(D(i).name);
    
    D(i).method = P.method;
    D(i).xDim   = P.xDim;
    D(i).cvf    = P.cvf;
    
    [tf, D(i).methodIdx] = ismember(D(i).method, allMethods);
  end
  % Only continue processing files that have test trials
  D = D([D.cvf]>0);

  if isempty(D)
    fprintf('ERROR: No valid files.  Exiting...\n');
    method = [];
    return;
  end
  
  for i = 1:length(D)    
    fprintf('Loading %s/%s...\n', runDir, D(i).name);
    ws = load(sprintf('%s/%s', runDir, D(i).name));
    
    % Check if selected kernSD has been run.
    if isfield(ws, 'kernSDList')
      % PCA, PPCA, FA
      [kernSD_exists, kidx] = ismember(kernSD, ws.kernSDList);
    else
      % GPFA
      kernSD_exists = true;
    end
    
    D(i).isValid = false;
    
    if kernSD_exists
      D(i).isValid = true;
      
      % Compute prediction error
      if D(i).methodIdx <= 3
        % PCA, PPCA, FA
        Ycs            = [ws.kern(kidx).seqTest.ycs];
        D(i).sse       = sum((Ycs(:)-ws.YtestRaw(:)).^2);            
        D(i).numTrials = length(ws.kern(kidx).seqTest);
      elseif D(i).methodIdx == 4
        % GPFA
        YtestRaw       = [ws.seqTest.y];
        for p = 1:D(i).xDim
          fn              = sprintf('ycsOrth%02d', p);
          Ycs             = [ws.seqTest.(fn)];
          D(i).sseOrth(p) = sum((Ycs(:) - YtestRaw(:)).^2); 
        end
        D(i).sse       = D(i).sseOrth(end);
        D(i).numTrials = length(ws.seqTest);        
      end
    end
  end
  
  D = D([D.isValid]);

  if isempty(D)
    fprintf('ERROR: No valid files.  Exiting...\n');
    method = [];
    return;
  end
  
  % Sum prediction error across cross-validation folds
  for n = 1:4
    Dn = D([D.methodIdx]==n);

    method(n).name = upper(allMethods{n});
    method(n).xDim = unique([Dn.xDim]);
    
    % Do for each unique state dimensionality.
    % Each method is allowed to have a different list of 
    % unique state dimensionalities.
    for p = 1:length(method(n).xDim)
      Dnn = Dn([Dn.xDim] == method(n).xDim(p));
      
      % Sum across cross-validation folds
      method(n).sse(p)       = sum([Dnn.sse]);        
      method(n).numTrials(p) = sum([Dnn.numTrials]);
    end
  end
  
  % Reduced GPFA (based on GPFA files with largest xDim)
  Dn = D([D.methodIdx]==4);

  if length(Dn)>0
    dList = [Dn.xDim];
    Dnn   = Dn(dList == max(dList));
    
    method(5).name      = 'Reduced GPFA';
    method(5).xDim      = 1:max(dList);
    method(5).sse       = sum(vertcat(Dnn.sseOrth));
    method(5).numTrials = sum([Dnn.numTrials]);
  end
    
  numTrialsAll = [method.numTrials];
  if length(unique(numTrialsAll)) ~= 1
    fprintf('ERROR: Number of test trials must be the same across\n');
    fprintf('all methods and state dimensionalities.  Exiting...\n');
    return
  end
    
  % =========
  % Plotting
  % =========
  if plotOn
    col = {'r--', 'r', 'g', 'k--', 'k'}; 
    
    figure;
    hold on;
    lgnd = {};
    for n = 1:length(method)
      if ~isempty(method(n).xDim)
        plot(method(n).xDim, method(n).sse, col{n});
        lgnd{end+1} = method(n).name;
      end
    end
    legend(lgnd{:});
    legend('boxoff');
    title(sprintf('For two-stage methods, kernel width = %d ms', kernSD),...
    'fontsize', 12);
    xlabel('State dimensionality', 'fontsize', 14);
    ylabel('Prediction error', 'fontsize', 14);
  end
