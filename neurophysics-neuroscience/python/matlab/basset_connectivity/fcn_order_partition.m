function [I,CI] = fcn_order_partition(x,ci,isrand)
% clear all
% close all
%
% load A
% x = A;
% ci = modularity_louvain_und(x,2.5);
nc = max(ci);
n = length(ci);
h = hist(ci,1:nc);

% [~,indsort] = sort(h,'descend');
I = zeros(n,1);
CI = zeros(n,1);
count = 0;
for i = 1:nc
    ind = ci == i;
    y = x(ind,ind);
    z = mean(y,2) + mean(y)';
    yy = x(ind,~ind);
    yy2 = x(~ind,ind);
    zz = mean(yy,2) + mean(yy2)';
    z = z - zz;
    vtx = find(ind);
    [~,jnd] = sort(z,'descend');
    if exist('isrand','var')
        jnd = jnd(randperm(length(jnd)))
    end
    vtx = vtx(jnd);
    
    I((count + 1):(count + length(jnd))) = vtx;
    CI((count + 1):(count + length(jnd))) = i;
    count = count + length(jnd);
end