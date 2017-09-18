function Pout = fillPerSymm(Pin, blkSize, T, varargin)
%
% Pout = fillPerSymm(Pin, blkSize, T,...)
%
% Fills in the bottom half of a block persymmetric matrix, given the
% top half.
%
% INPUTS:
%
% Pin      - top half of block persymmetric matrix
%            (xDim*Thalf) x (xDim*T), where Thalf = ceil(T/2)
% blkSize  - edge length of one block
% T        - number of blocks making up a row of Pin
%
% OUTPUTS:
%
% Pout     - full block persymmetric matrix
%            (xDim*T) x (xDim*T)
%
% OPTIONAL ARGUMENTS:
%
% blkSizeVert - vertical block edge length if blocks are not square.
%               'blkSize' is assumed to be the horizontal block edge
%               length.
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu

  blkSizeVert = blkSize;
  assignopts(who, varargin);
    
  % Fill in bottom half by doing blockwise fliplr and flipud
  Thalf   = floor(T/2);
  idxHalf = bsxfun(@plus, (1:blkSizeVert)', ((Thalf-1):-1:0)*blkSizeVert);
  idxFull = bsxfun(@plus, (1:blkSize)', ((T-1):-1:0)*blkSize);
    
  Pout = [Pin; Pin(idxHalf(:), idxFull(:))];
