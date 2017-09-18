function [abspos, relpos] = nestplotpos(pp, varargin)
% NESTPPLOTPOS - position vector for nestplot
%
% NESTPPLOTPOS(P) returns the position vector for the Pth nestplot
% in the current grid.
%
% NESTPPLOTPOS(M,N,P) or NESTPPLOTPOS(P, 'grid', [M,N]) returns the
% position vector for the Pth nestplot in an MxN grid within the
% current container.  It does not change the container grid.
%
% NESTPPLOTPOS(..., 'container', AX) acts on the specified
% container. 

% maneesh
% 20120704: created.
% 20131018: fixed spacing bug in multicell coordinates
% 20140501: added order option

container = [];    % [gca] container axes
spacing = [];
xspacing = [];
yspacing = [];
nochecks = 0;
order = '';
grid = [];

if nargin >= 3 && isnumeric(varargin{1})
  grid = [pp, varargin{1}];
  pp = varargin{2};
  varargin(1:2) = [];
end
  
optlistwarn(optlistassign(who, varargin));

if isempty(container)
  container = gca;
end

pdat = getappdata(container, 'NestData');

if (isfield(pdat, 'nestable'))
  while pdat.nestable == 0
    container = pdat.container;
    pdat = getappdata(container, 'NestData');
  end
end

if isempty(spacing)
  spacing = pdat.spacing;
end

if isempty(xspacing)
  xspacing = spacing(1);
end

if isempty(yspacing)
  yspacing = spacing(end);
end

if isempty(grid)
  grid = pdat.grid;
  if isempty(grid)
    error('No grid specified and none established.');
  end
end

if isempty(order)
  order = pdat.gridorder;
end

oldunits = get(container, 'Units');
set (container, 'Units', 'normalized');
ppos = get(container, 'Position');
set (container, 'Units', oldunits);

gridbox = ppos(3:4) ./ fliplr(grid);

if iscell(pp)
  ii = pp{1}-1;
  jj = pp{2}-1;
else
  switch order
    case {'rowfirst', 'row'}
     ii = floor((pp-1)/grid(2));
     jj = mod  ((pp-1),grid(2));
   case {'colfirst', 'columnfirst', 'col', 'column'}
     ii = mod  ((pp-1),grid(1));
     jj = floor((pp-1)/grid(1));
  end
end

if ~nochecks && (any(ii >= grid(1)) || any(jj >= grid(2)))
  if iscell(pp)
    error(sprintf('Grid [%d,%d] too small for [%s, %s]', grid, num2str(ii+1),num2str(jj+1)));
  else
    error(sprintf('Grid [%d,%d] too small for %d', grid, p));
  end
end

abspos(1:2) = [min(jj), min(ii)] .* gridbox;
abssiz(1:2) = [1+range(jj), 1+range(ii)].*gridbox;

abspos(1)   = ppos(1) + abspos(1) + xspacing/2 * gridbox(1);
abspos(2)   = ppos(2) + ppos(4) - abspos(2) - abssiz(2) + yspacing/2 * gridbox(2);

abspos(3:4) = abssiz - [xspacing,yspacing].*gridbox;

if nargout > 1
  relpos = [(abspos(1:2) - ppos(1:2))./ppos(3:4), abspos(3:4)./ ppos(3:4)];
end