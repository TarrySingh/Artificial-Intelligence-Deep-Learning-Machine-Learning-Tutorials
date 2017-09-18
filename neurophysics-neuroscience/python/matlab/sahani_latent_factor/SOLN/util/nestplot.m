function hhout = nestplot(m,n,pvec,varargin)
% nestplot - powerful, flexible, nested subplots: h=nestplot(m,n,pp,...)
%
% NESTPLOT(M, N, P, ...) creates one or more new axes within the
% region defined by the current axes.  It sets the container axes
% visibility off (but leaving the axis labels and title, if any,
% visible), makes the new axes current and returns a handle to them.
% M and N set the number of columns and rows of the nest grid.  P
% specifies which axes to make.  It may be a scalar or vector,
% indexing plots left to right, top to bottom.  Or it may be a row-col
% reference cell array of the form {R, C} or {RMIN:RMAX, CMIN:CMAX}
% which specified axes that spans all the given slots.  Multiple
% row-col references may be passed within a containing cell array:
% {{R1,C1}, {R2,C2}, ...}.  If P is empty no plots are created but
% the grid is established within the container.
%
% NESTPLOT([], [], P, ...) or NESTPLOT(P, ...) uses the current
% nest grid size (and generates an error if there is none).
%
% NESTPLOT(Inf) or NESTPLOT('next') creates and selects the first
% non-existent nestplot in the current grid.  It generates an error if
% there is no grid, or if it is fully populated.
%
% NESTPLOT(Inf, Inf, P, ...) or NESTPLOT('auto', P, ...) resets the
% grid if needed to accomodate the plots specified in P, which must be
% in row-column form.  Existing nestplots are repositioned according
% to their grid coordinates.
%
% NESTPLOT(AX, ...) or NESTPLOT(..., 'container', AX) uses the
% specified axes as a container, rather than the current one.  If AX
% (or the current axes, if AX is not given) are themselves nested,
% then nestplot creates a peer within the corresponding container.
% This behaviour can be turned off for a nested plot by calling
% NESTABLE, thus allowing hierarchical nesting.
%
% If NESTPLOT is called twice with the same parameters in the same
% axes it recovers the handle to the previous axes instead of creating
% a new one.
%
% H = NESTPLOT(...) returns (a vector of) handles to nested axes.
%
% OPTIONS:
% 'spacing'	[.1]	fraction of child dimension between adjacent siblings
% 'xspacing'	[.1]	fraction of child width between adjacent siblings
% 'yspacing'	[.1]	fraction of child height between adjacent siblings
% 'killoverlap'	[1]	delete overlapping but nonidentical axes
% 'reuseaxes'	[]	move these axes into nest rather than new ones
% 'container'	[gca]	container axes
% 'order'	[row]	[row|col]first sequence for scalar numbering
%
% See also: NESTABLE, NESTGROUP.

% maneesh.
% pre-20020324: created
% 20130702: doc cleanup
% 20130704: added 'spacing'
% 20130704: switched data store from userdata to appdata
% 20130704: merged ability to span multiple slots from gridplot
% 20130727: default killoverlap=0 if any(spacing<0)
% 20140501: use nestable() to find container of gca
%     added order option
% 20140520: added option to reuse current grid params
% 20141001: added 'next' and empty P behaviour
% 20150616: don't reset position of matched axes
% 20150819: added 'auto' behaviour

% OPTIONS:
spacing = [];      % [.1] fraction of child dimension between adjacent siblings
xspacing = [];     % [.1] fraction of child width between adjacent siblings
yspacing = [];	   % [.1] fraction of child height between adjacent siblings
killoverlap = [];  % [1] delete overlapping but nonidentical axes
reuseaxes = [];    % [] move these axes into nest rather than new ones
container = [];    % [gca] container axes
order = 'rowfirst';% [row] [row|col]first sequence for scalar numbering
% OPTIONS DONE

nargs = nargin;

% check for NESTPLOT(AX, ...)
if prod(size(m))==1 && ishandle(m) && strcmp(get(m,'type'), 'axes')
  switch (nargs)
   case 1,
    error('nothing to do');
   case 2, 
    [container,m] = deal(m,n);
   case 3, 
    [container,m,n] = deal(m,n,pvec);
   otherwise, 
    [container,m,n,pvec] = deal(m,n,pvec,varargin{1});
    varargin(1) = [];
  end    
  nargs = nargs - 1;
end


if nargs == 1                           % nestplot(P)
  [m,n,pvec] = deal([],[],m);
elseif (ischar(m) && strcmp(m, 'auto')) % nestplot('auto', P, ..)
  if ~iscell(n)
    error('must specify plot as {p,q} for auto mode');
  end
  [m,n,pvec] = deal(Inf, Inf, n);
  if (nargin > 2) 
    varargin = {pvec varargin{:}};
  end
elseif nargs == 2                       % nestplot(M,N)    
  pvec = [];                          
end

if ischar(n)                            % nestplot(P,'opt1',...)
  varargin = {n pvec varargin{:}};
  [m,n,pvec] = deal([],[],m);
end  

if ischar(pvec)
  if strcmp(pvec, 'next')
    pvec = Inf;
  else                                    % nestplot(M,N,'opt1',...)
    varargin = {pvec, varargin{:}};
    pvec = [];
  end
end
  
axargs = optlistassign(who, varargin);

if isempty(container)
  container = nestable(gca);
end
pdat = getappdata(container, 'NestData');

if isempty(spacing)
  if isfield(pdat, 'spacing')
    spacing = pdat.spacing;
  else
    spacing = [.1,.1];                       % default value
  end
end
if isempty(xspacing) xspacing = spacing(1);   end
if isempty(yspacing) yspacing = spacing(end); end

if isempty(killoverlap)
  if xspacing < 0 || yspacing < 0
    killoverlap = 0;
  else
    killoverlap = 1;
  end
end

if (isempty(pdat))
  % new container
  if (isempty(m) || isempty(n))
    error('no established nestplot grid');
  end
  c_hand = handle(container);
  c_posp = findprop(c_hand,'Position');
  listener = addlistener(c_hand, 'Position', 'PostSet',...
                             @(x,y) nestresize(container));
  pdat = struct(...
      'grid', [1, 1], ...
      'nestable', 1, ...
      'container', [], ...
      'children', [], ...
      'resizelistener', listener);
  setappdata(container, 'NestData', pdat);
end

if ((~isempty(m) && isinf(m)) || (~isempty(n) && isinf(n)))
  if ~iscell(pvec{1})
    pvecm = max(pvec{1}); pvecn = max(pvec{2});
  else
    pvecm = cellfun(@(x) max(x{1}), pvec, 'UniformOutput', 1);
    pvecn = cellfun(@(x) max(x{2}), pvec, 'UniformOutput', 1);
  end
  if isempty(m) || isinf(m) m = max(pvecm); end
  if isempty(n) || isinf(n) n = max(pvecn); end
  if m > pdat.grid(1) || n > pdat.grid(2)
    nestgroup('regrid', max(m, pdat.grid(1)), max(n, pdat.grid(2)));
  end
end
      
% update grid and spacing information for position calculation
if ~isempty(m) pdat.grid(1) = m; end
if ~isempty(n) pdat.grid(2) = n; end
pdat.spacing = [xspacing, yspacing];
pdat.gridorder = order;

setappdata(container, 'NestData', pdat);

outaxs = [];

if iscell(pvec) && ~iscell(pvec{1})
  % protect single-level cell array from expansion by FOR
  pvec = {pvec};
end


if isnumeric(pvec) && any(isinf(pvec))

  if (isempty(pdat.children))
    avail = 1:prod(pdat.grid);
  else
    for cc=1:length(pdat.children)
      if ishandle(pdat.children(cc)) && ...
            strcmp(get(pdat.children(cc), 'type'), 'axes')
        peers(cc) = getfield(getappdata(pdat.children(cc), 'NestData'),...
                             'gridposition');
      else
        peers(cc) = NaN;
      end
    end
    avail = setdiff(1:prod(pdat.grid), peers);
  end

  ninfs = nnz(isinf(pvec));
  if (ninfs > length(avail))
    error('grid either nonexistant or too small for next nestplot');
  end
  pvec(isinf(pvec)) = avail(1:ninfs);
end
  

iChild = 0;
for p = pvec

  iChild = iChild + 1;
  % FOR on a cell array returns a singleton sub-array, not a member
  if iscell(p)
    p = p{1};
  end

  newaxs = [];
  cdat = [];

  % Were we given a list of axes objects to move into the nest?
  if (~isempty(reuseaxes))
    newaxs = reuseaxes(iChild);
    if ~ishandle(newaxs) || ~strcmp(get(newaxs, 'type'), 'axes')
      % list might contain 0s -- skip those
      newaxs = 0;
      outaxs = [outaxs ; newaxs];
      continue;
    end
    cdat = getappdata(newaxs, 'NestData');
  end

  if ~isempty(newaxs) && ~isempty(cdat.gridposition)
    [newpos, relpos] = nestplotpos(pdat.grid(1), pdat.grid(2), ...
                                   cdat.gridposition, ...
                                   'container', container);
  else
    [newpos, relpos] = nestplotpos(pdat.grid(1), pdat.grid(2),...
                                   p, ...
                                   'container', container);
  end

  if ~isempty(newaxs)
    axes(newaxs);
    set(newaxs, 'Position', newpos, axargs{:});
  else
    % look for existing axes in the right place
    tol = sqrt(eps);
    sibs = get(gcf, 'Children');
    for i = 1:length(sibs)
      % check that sibs(i) is still alive, and not a current nest container
      if (ishandle(sibs(i)) && strcmp(get(sibs(i),'Type'),'axes') && ...
         sibs(i) ~= container && ...
          ~(nestable(sibs(i)) == sibs(i) && ~isempty(getappdata(sibs(i), 'NestData'))));

        oldunits = get(sibs(i),'Units');
        set(sibs(i),'Units','normalized')
        sibpos = get(sibs(i),'Position');
        set(sibs(i),'Units',oldunits);

        intersect = 1;
        if((newpos(1) >= sibpos(1) + sibpos(3)-tol) | ...
           (sibpos(1) >= newpos(1) + newpos(3)-tol) | ...
           (newpos(2) >= sibpos(2) + sibpos(4)-tol) | ...
           (sibpos(2) >= newpos(2) + newpos(4)-tol))
          intersect = 0;
        end
        
        if intersect
          if any(abs(sibpos - newpos) > tol)
            if killoverlap
              delete(sibs(i));
            end
          else
            newaxs = sibs(i);
            axes(newaxs);
            if length(axargs) set(newaxs, axargs{:}); end
          end
        end
      end
    end  
  end

  if (isempty(newaxs))
    % didn't find any: make a new one
    newaxs = axes;
    set(newaxs, 'Position', newpos, axargs{:});
  end

  if isempty(cdat)
    cdat = struct(...
        'nestable', 0, ...
        'gridposition', {p});
  end
  
  cdat.container = container;
  cdat.position = relpos;

  setappdata (newaxs, 'NestData', cdat);
  
  set(newaxs, 'DeleteFcn', @nestdelete);

  
  if strcmp(get(newaxs, 'NextPlot'), 'replace')
    set(newaxs, 'NextPlot', 'replacechildren');
  end  

  outaxs = [outaxs ; newaxs];

end


  
set (container, 'Visible', 'off');
set (get(container, 'xlabel'), 'Visible', 'on');
set (get(container, 'ylabel'), 'Visible', 'on');
set (get(container, 'zlabel'), 'Visible', 'on');
set (get(container, 'title'), 'Visible', 'on');

pdat = getappdata(container, 'NestData');
if isnumeric(pvec)
  pdat.children(pvec) = outaxs;
else
  pdat.children(end+(1:length(outaxs))) = outaxs;
end
setappdata (container, 'NestData', pdat);

if (nargout)
  hhout = outaxs;
end



%-----------------------------------------------------------------------
function nestdelete(ax, eventdata)

ndat = getappdata(ax,'NestData');
if ishandle(ndat.container)                % might have been deleted
  pdat = getappdata(ndat.container, 'NestData');
  pdat.children(find(pdat.children == ax)) = 0;
  setappdata(ndat.container, 'NestData', pdat);
end


