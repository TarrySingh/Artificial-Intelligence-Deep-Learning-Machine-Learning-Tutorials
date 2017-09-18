function [X,Y,ind] = fcn_plot_blocks(g)
%FCN_PLOT_BLOCKS    visualize block structure
%
%   [X,Y,IND] = FCN_PLOT_BLOCKS is a visualization tool for plotting block
%               model boundaries. As input it takes a vec G of size N x 1
%               (N is the number of nodes) where each cell is the group
%               assignment for the corresponding node. As an output, it
%               returns X and Y, which are vectors of boundaries and also
%               IND, which is the ordering of the adjacency matrix to fit
%               with the block structure.
%
%   Example:    >> [ci,q] = modularity_louvain_und(A); % find modules
%               >> [x,y,ind] = fcn_plot_blocks(ci);    % get vectors
%               >> imagesc(A(ind,ind)); hold on; plot(x,y,'w');
%
%   Inputs:     G,      vector of group assignments
%
%   Outputs:    X,      x-coordinates
%               Y,      y-coordinates
%               IND,    reordering for adjacency matrix
%
%   Richard Betzel, Indiana University, 2013
%

nc = max(g);
n = length(g);
x = 0.5;
y = n + 0.5;
[gsort,ind] = sort(g);
X = [];
Y = [];
for i = 2:(nc)
    aa = find(gsort == i);
    xx = [x; y];
    yy = (aa(1) - 0.5)*ones(2,1);
    X = [X; xx; NaN; yy; NaN];
    Y = [Y; yy; NaN; xx; NaN];
end