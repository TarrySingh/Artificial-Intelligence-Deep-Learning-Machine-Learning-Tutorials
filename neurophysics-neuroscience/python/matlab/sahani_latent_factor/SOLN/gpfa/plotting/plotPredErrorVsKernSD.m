function method = plotPredErrorVsKernSD(runIdx, xDim, varargin)
%
% method = plotPredErrorVsKernSD(runIdx, xDim,...)
%
% Plot prediction error versus smoothing kernel standard deviation.
%
% INPUTS:
%
% runIdx    - results files will be loaded from mat_results/runXXX, where
%             XXX is runIdx
% xDim      - state dimensionality to be plotted for all methods
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
  % File has test data
  crit1 = ([D.cvf] > 0);
  % File uses selected state dimensionality
  crit2 = ([D.xDim] == xDim);
  % File can be potentially used for reduced GPFA, which will be
  % based on GPFA files with largest xDim.
  crit3 = ([D.xDim] > xDim) & ([D.methodIdx] == 4);  
  % Only continue processing files that satisfy criteria
  D = D(crit1 & (crit2 | crit3));

  if isempty(D) || isempty(intersect([D.methodIdx], [1 2 3]))
    fprintf('ERROR: No valid two-stage files.  Exiting...\n');
    method = [];
    return
  end
  
  for i = 1:length(D)   
    fprintf('Loading %s/%s...\n', runDir, D(i).name);
    ws = load(sprintf('%s/%s', runDir, D(i).name));
    
    % Compute prediction error
    if D(i).methodIdx <= 3
      % PCA, PPCA, FA
      sse = nan(size(ws.kern));
      for k = 1:length(sse)          
        Ycs    = [ws.kern(k).seqTest.ycs];
        sse(k) = sum((Ycs(:)-ws.YtestRaw(:)).^2);            
      end
      D(i).kernSD    = ws.kernSDList;
      D(i).sse       = sse;
      D(i).numTrials = length(ws.kern(1).seqTest);
      
    elseif D(i).methodIdx == 4
      % GPFA
      YtestRaw       = [ws.seqTest.y];
      fn             = sprintf('ycsOrth%02d', xDim);
      Ycs            = [ws.seqTest.(fn)];
      D(i).sse       = sum((Ycs(:) - YtestRaw(:)).^2); 
      D(i).kernSD    = NaN;
      D(i).numTrials = length(ws.seqTest);
    end
  end
    
  % Sum prediction error across cross-validation folds
  for n = 1:4
    method(n).name = upper(allMethods{n});
    
    Dn = D(([D.methodIdx]==n) & ([D.xDim]==xDim));
    if isempty(Dn)
      continue
    end

    % Ensure that same kernelSD were used in each fold
    for i = 1:length(Dn)
      if ~isequalwithequalnans(Dn(i).kernSD, Dn(1).kernSD)
        fprintf('ERROR: kernSD used in each cross-validation fold\n');
        fprintf('must be identical.  Exiting...\n');
        return
      end
    end
    
    method(n).kernSD    = Dn(1).kernSD;
    method(n).sse       = sum(vertcat(Dn.sse));
    method(n).numTrials = sum([Dn.numTrials]);
  end
  
  % Reduced GPFA (based on GPFA files with largest xDim)
  Dn = D([D.methodIdx]==4);
  if length(Dn) > 0
    dList = [Dn.xDim];
    Dnn   = Dn(dList == max(dList));
    
    method(5).name      = 'Reduced GPFA';
    method(5).kernSD    = NaN;
    method(5).sse       = sum([Dnn.sse]);
    method(5).numTrials = sum([Dnn.numTrials]);
  end
    
  numTrialsAll = [method.numTrials];
  if length(unique(numTrialsAll)) ~= 1
    fprintf('ERROR: Number of test trials must be the same across\n');
    fprintf('all methods.  Exiting...\n');
    return
  end
  
  % GPFA prediction error does not depend on kernSD, but put in same
  % format as the other methods for ease of plotting.
  minKernSD = min([method.kernSD]);
  maxKernSD = max([method.kernSD]);
  for n = 1:length(method)
    if isnan(method(n).kernSD)
      method(n).kernSD = [minKernSD maxKernSD];
      method(n).sse    = method(n).sse * [1 1];
    end
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
      if ~isempty(method(n).kernSD)
        plot(method(n).kernSD, method(n).sse, col{n});
        lgnd{end+1} = method(n).name;
      end
    end
    legend(lgnd{:});
    legend('boxoff');
    title(sprintf('State dimensionality = %d', xDim), 'fontsize', 12);
    xlabel('Kernel width (ms)', 'fontsize', 14);
    ylabel('Prediction error', 'fontsize', 14);
  end
