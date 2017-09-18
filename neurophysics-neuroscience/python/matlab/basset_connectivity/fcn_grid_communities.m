function [X Y indsort] = fcn_grid_communities(c)
%FCN_GRID_COMMUNITIES       outline communities along diagonal
%
%   [X Y INDSORT] = FCN_GRID_COMMUNITIES(C) returns {X,Y} coordinates for
%                   highlighting modules located along the diagonal.
%                   INDSORT are the indices to sort the matrix.
%
%   Inputs:     C,       community assignments
%
%   Outputs:    X,       x coor
%               Y,       y coor
%               INDSORT, indices
%
%   Richard Betzel, Indiana University, 2012
%

nc = max(c);
[c,indsort] = sort(c);

X = [];
Y = [];
for i = 1:nc
    ind = find(c == i);
    if ~isempty(ind)
        mn = min(ind) - 0.5;
        mx = max(ind) + 0.5;
        x = [mn mn mx mx mn NaN];
        y = [mn mx mx mn mn NaN];
        X = [X, x];
        Y = [Y, y];
    end
end