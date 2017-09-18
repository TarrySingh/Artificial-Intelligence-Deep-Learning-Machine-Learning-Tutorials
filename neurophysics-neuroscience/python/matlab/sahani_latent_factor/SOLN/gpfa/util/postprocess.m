function [estParams, seqTrain, seqTest] = postprocess(ws, varargin)
%
% [estParams, seqTrain, seqTest] = postprocess(ws, ...)
%
% Orthonormalization and other cleanup.
%
% INPUT:
%
% ws        - workspace variables returned by neuralTraj.m
%
% OUTPUTS:
%
% estParams - estimated model parameters, including 'Corth' obtained
%             by orthonormalizing the columns of C
% seqTrain  - training data structure containing new field 'xorth', 
%             the orthonormalized neural trajectories
% seqTest   - test data structure containing orthonormalized neural
%             trajectories in 'xorth', obtained using 'estParams'
%
% OPTIONAL ARGUMENT:
%
% kernSD    - for two-stage methods, this function returns seqTrain
%             and estParams corresponding to kernSD.  By default, 
%             the function uses ws.kern(1).
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  kernSD = [];
  assignopts(who, varargin);

  estParams = [];
  seqTrain  = [];
  seqTest   = [];

  if isempty(ws)
    fprintf('ERROR: Input argument is empty.\n');
    return
  end

  if isfield(ws, 'kern') 
    if isempty(ws.kernSDList)
      k = 1;
    else
      k = find(ws.kernSDList == kernSD);
      if isempty(k)
        fprintf('ERROR: Selected kernSD not found.\n');
        return
      end
    end
  end

  if ismember(ws.method, {'gpfa'})
    C               = ws.estParams.C;
    X               = [ws.seqTrain.xsm];    
    [Xorth, Corth]  = orthogonalize(X, C); 
    seqTrain        = segmentByTrial(ws.seqTrain, Xorth, 'xorth');
    
    estParams       = ws.estParams;
    estParams.Corth = Corth;

    if ~isempty(ws.seqTest)
      fprintf('Extracting neural trajectories for test data...\n');
      
      ws.seqTest     = exactInferenceWithLL(ws.seqTest, estParams);
      X              = [ws.seqTest.xsm];
      [Xorth, Corth] = orthogonalize(X, C);
      seqTest        = segmentByTrial(ws.seqTest, Xorth, 'xorth');            
    end
  
  elseif ismember(ws.method, {'fa', 'ppca'})
    L               = ws.kern(k).estParams.L;
    X               = [ws.kern(k).seqTrain.xpost];
    [Xorth, Lorth]  = orthogonalize(X, L); 
    seqTrain        = segmentByTrial(ws.kern(k).seqTrain, Xorth, 'xorth');    
    
    % Convert to GPFA naming/formatting conventions
    estParams.C     = ws.kern(k).estParams.L;
    estParams.d     = ws.kern(k).estParams.d;
    estParams.Corth = Lorth;
    estParams.R     = diag(ws.kern(k).estParams.Ph);

    if ~isempty(ws.kern(k).seqTest)
      fprintf('Extracting neural trajectories for test data...\n');
      
      Y              = [ws.kern(k).seqTest.y];
      X              = fastfa_estep(Y, ws.kern(k).estParams);
      [Xorth, Lorth] = orthogonalize(X.mean, L);
      seqTest        = segmentByTrial(ws.kern(k).seqTest, Xorth, 'xorth');    
    end
        
  elseif ismember(ws.method, {'pca'})
    % PCA is already orthonormalized
    X               = [ws.kern(k).seqTrain.xpost];
    seqTrain        = segmentByTrial(ws.kern(k).seqTrain, X, 'xorth');
    
    estParams.Corth = ws.kern(k).estParams.L;
    estParams.d     = ws.kern(k).estParams.d;
    
    if ~isempty(ws.kern(k).seqTest)
      fprintf('Extracting neural trajectories for test data...\n');
      
      Y       = [ws.kern(k).seqTest.y];
      Xorth   = estParams.Corth' * bsxfun(@minus, Y, estParams.d);
      seqTest = segmentByTrial(ws.kern(k).seqTest, Xorth, 'xorth');
    end    
    
  else
    fprintf('ERROR: method not recognized.\n');
  end
