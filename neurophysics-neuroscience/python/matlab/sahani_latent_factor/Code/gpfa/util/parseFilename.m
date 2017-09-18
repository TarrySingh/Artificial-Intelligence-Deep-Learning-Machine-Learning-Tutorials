function [result, err] = parseFilename(str)
%
% [result, err] = parseFilename(str)
%
% Extracts method, xDim and cross-validation fold from results filename,
% where filename has format [method]_xDim[xDim]_cv[cvf].mat.
%
% INPUTS:
%
% str      - filename string to parse
%
% OUTPUTS:
%
% result   - structure with fields method, xDim, and cvf
% err      - boolean that indicates if input string is invalid filename
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  result = [];
  err    = false;
  
  undi = find(str == '_');
  if isempty(undi)   
    err    = true;
    return
  end
  
  result.method = str(1:undi(1)-1);    
  
  [A, count, errmsg] = sscanf(str(undi(1)+1:end), 'xDim%d_cv%d.mat');
    
  if (count < 1) || (count > 2)
    err = true;
    return
  end
  
  result.xDim = A(1);
  if count == 1
    result.cvf = 0;
  else
    result.cvf = A(2);
  end
