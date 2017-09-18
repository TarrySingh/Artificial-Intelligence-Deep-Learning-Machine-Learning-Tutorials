function [invM, logdet_M] = invPerSymm(M, blkSize, varargin)
%
% [invM, logdet_M] = invPerSymm(M, blkSize,...)
%
% Inverts a matrix that is block persymmetric.  This function is
% faster than calling inv(M) directly because it only computes the
% top half of inv(M).  The bottom half of inv(M) is made up of
% elements from the top half of inv(M).
%
% WARNING: If the input matrix M is not block persymmetric, no
% error message will be produced and the output of this function will
% not be meaningful.
%
% INPUTS:
%
% M        - the block persymmetric matrix to be inverted
%            ((blkSize*T) x (blkSize*T)).  Each block is 
%            blkSize x blkSize, arranged in a T x T grid.
% blkSize  - edge length of one block
%
% OUTPUTS:
%
% invM     - inverse of M ((blkSize*T) x (blkSize*T))
% logdet_M - log determinant of M
%
% OPTIONAL ARGUMENTS:
%
% offDiagSparse - logical that specifies whether off-diagonal blocks are
%                 sparse (default: false)
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu
  
  offDiagSparse = 'false'; % specify if A12 is sparse
  assignopts(who, varargin);

  T     = size(M, 1) / blkSize;  
  Thalf = ceil(T/2);
  mkr   = blkSize * Thalf;

  invA11 = inv(M(1:mkr, 1:mkr));
  invA11 = (invA11 + invA11') / 2;
  
  if offDiagSparse
    A12 = sparse(M(1:mkr, (mkr+1):end));
  else
    A12 = M(1:mkr, (mkr+1):end);
  end

  term   = invA11 * A12;
  F22    = M(mkr+1:end, mkr+1:end) - A12' * term;
  
  res12  = -term / F22;
  res11  = invA11 - res12 * term';
  res11  = (res11 + res11') / 2;
  
  % Fill in bottom half of invM by picking elements from res11 and res12
  invM = fillPerSymm([res11 res12], blkSize, T);
  
  if nargout == 2
    logdet_M = -logdet(invA11) + logdet(F22);
  end
