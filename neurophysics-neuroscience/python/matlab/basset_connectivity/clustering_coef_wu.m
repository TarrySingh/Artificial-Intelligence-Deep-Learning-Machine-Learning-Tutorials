function C=clustering_coef_wu(W)
%CLUSTERING_COEF_WU     Clustering coefficient
%
%   C = clustering_coef_wu(W);
%
%   The weighted clustering coefficient is the average "intensity"
%   (geometric mean) of all triangles associated with each node.
%
%   Input:      W,      weighted undirected connection matrix
%                       (all weights must be between 0 and 1)
%
%   Output:     C,      clustering coefficient vector
%
%   Note:   All weights must be between 0 and 1.
%           This may be achieved using the weight_conversion.m function,
%           W_nrm = weight_conversion(W, 'normalize');
%
%   Reference: Onnela et al. (2005) Phys Rev E 71:065103
%
%
%   Mika Rubinov, UNSW/U Cambridge, 2007-2015

%   Modification history:
%   2007: original
%   2015: expanded documentation

K=sum(W~=0,2);            	
cyc3=diag((W.^(1/3))^3);           
K(cyc3==0)=inf;             %if no 3-cycles exist, make C=0 (via K=inf)
C=cyc3./(K.*(K-1));         %clustering coefficient
