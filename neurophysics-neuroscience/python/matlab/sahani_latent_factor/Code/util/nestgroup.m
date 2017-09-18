function h = nestgroup(ax, varargin)
% nestgroup - nest and manipulate existing axes: H=nestgroup([AX])
%
% NESTGROUP returns a handle to the container of the current axes.  If
% there is none it just returns the current axes, creating them if
% needed.  Thus commands like TITLE(NESTGROUP, 'title') can be used to
% set a supertitle for the nestgroup.
%
% NESTGROUP('position', POS) sets the position of the nestgroup
% container witin the figure.  POS maybe a MATLAB position vector,
% or any of the keywords 'max','full','fig', '0011', '111', which
% all correspond to [0 0 1 1].
% 
% NESTGROUP FULLFIGURE or FULLFIG is shorthand for
% NESTGROUP('position', [0, 0, 1, 1]);
% 
% NESTGROUP PEERS or NESTGROUP('peers') returns the handles of all
% existing peer axes in the nestgroup.
%
% NESTGROUP(AX) where AX is a vector of handles creates a new
% (invisible) container axes object that bounds all the specified
% axes, which are then nested within the new container.  It returns a
% handle to the new container.
%
% NESTGROUP ALL or NESTGROUP('all') groups all existing axes in the
% current figure. 
%
% NESTGROUP('regrid', M, N) resets the grid layout of an existing set
% of nestplots (that includes the current axes) to M x N.  The
% existing plots are moved to their corresponding positions in the new
% grid using the 'reuseaxes' option to NESTPLOT.
%
% NESTGROUP [ADDROW|ADDCOL[UMN]] adds a row or column to the
% current grid.
%
% See also: NESTPLOT, NESTABLE

% maneesh.
% 20130704: created
% 20130727: added peers option
% 20140424: added position option
% 20140522: fix 'nestgroup all' for only one current axes.


% OPTIONS:
padding = 0;   % expand container by this fraction of width/height
xpadding = []; % [0] expand container horizontally by this fraction of width
ypadding = []; % [0] expand container vertically by this fraction of height
% OPTIONS:done

if nargin == 0
  h = nestable(gca);
  if h == gca
    mydat = getappdata(h,'NestData');
    if isfield(mydat, 'container')
      h = mydat.container;
    end
  end
  return;
end

if (ischar(ax))
  switch lower(ax)
   case {'fullfigure', 'fullfig'},
    h = nestable(gca);
    if h == gca
      mydat = getappdata(h,'NestData');
      if isfield(mydat, 'container')
        h = mydat.container;
      end
    end
    set(h, 'position', [0 0 1 1]);
    return;
    
   case 'position', 
    h = nestable(gca);
    if h == gca
      mydat = getappdata(h,'NestData');
      if isfield(mydat, 'container')
        h = mydat.container;
      end
    end
    if (ischar(varargin{1}))
      switch(lower(varargin{1}))
       case {'max','full','fig', 'fullfigure', '0011', '111'}, 
        set(h, 'position', [0 0 1 1]);
       otherwise,
        error('unrecognised position argument');
      end
    else
      set(h, 'position', varargin{1});
    end
    return;
    
    
   case 'peers', 
    prnt = nestable(gca);
    pdat = getappdata(prnt,'NestData');
    h = pdat.children;
    return;
    
   case 'all',
    ax = findobj(gcf, 'type', 'axes');

   case 'addrow',
    prnt = nestable(gca);
    pdat = getappdata(prnt,'NestData');
    if isempty(pdat.grid) error('unknown grid: use ''regrid''.'); end
    nestplot(pdat.grid(1)+1, pdat.grid(2), 1:length(pdat.children), ...
             'reuseaxes', pdat.children);
    return;
    
   case {'addcol', 'addcolumn'}
    prnt = nestable(gca);
    pdat = getappdata(prnt,'NestData');
    if isempty(pdat.grid) error('unknown grid: use ''regrid''.'); end
    nestplot(pdat.grid(1), pdat.grid(2)+1, 1:length(pdat.children), ...
             'reuseaxes', pdat.children);
    return;
    
   case 'regrid', 
    mm = varargin{1};
    nn = varargin{2};
    prnt = nestable(gca);
    pdat = getappdata(prnt,'NestData');
    nestplot(mm, nn, 1:length(pdat.children), 'reuseaxes', pdat.children);
    return;
  end
end

axopts = optlistassign(who, varargin);

if isempty(xpadding) xpadding = padding(1); end
if isempty(ypadding) ypadding = padding(end); end

pos = get(ax, 'position');
if (iscell(pos))
  pos = cat(1, pos{:});
end

bounds = [min(pos(:,1:2)) max(pos(:,3:4)+pos(:,1:2))];
xpad = xpadding*bounds(3);
ypad = ypadding*bounds(3);

bounds = bounds + [-xpad, -ypad, 2*xpad, 2*ypad];
bounds = min(1, max(0, bounds));

parpos = [bounds(1:2), bounds(3:4)-bounds(1:2)];

hh = axes('position', parpos, axopts{:});
set (hh, 'Visible', 'off');
set (get(hh, 'xlabel'), 'Visible', 'on');
set (get(hh, 'ylabel'), 'Visible', 'on');
set (get(hh, 'zlabel'), 'Visible', 'on');
set (get(hh, 'title'), 'Visible', 'on');

for aa = 1:length(ax)
  setappdata (ax(aa), 'NestData', struct(...
      'nestable', 0, ...
      'container', hh, ...
      'gridposition', [], ...
      'position', [(pos(aa,1:2)-parpos(1:2))./parpos(3:4), ...
                   pos(aa,3:4)./parpos(3:4)]...
      ));
  set(ax(aa), 'DeleteFcn', @nestdelete);
end

hh_hand = handle(hh);
hh_ppos = findprop(hh_hand,'Position');
listener = handle.listener(hh_hand, hh_ppos,'PropertyPostSet',...
                           @(x,y) nestresize(hh));
  
setappdata (hh, 'NestData', struct(...
    'grid', [], ...
    'nestable', 1, ...
    'container', [], ...
    'children', ax, ...
    'resizelistener', listener));



%-----------------------------------------------------------------------
function nestdelete(ax)

ndat = getappdata(ax,'NestData');
if ishandle(ndat.container)                % might have been deleted
  pdat = getappdata(ndat.container, 'NestData');
  pdat.children(find(pdat.children == ax)) = 0;
  setappdata(ndat.container, 'NestData', pdat);
end

