function h = fig(id, varargin)
% FIG - find or create a figure: [h=]fig('name')
%
% FIG and FIG(H) where H is a figure handle are almost identical to
% the built-in FIGURE, expect that if a new figure is created the
% name may include a prefix.
%
% FIG('name') looks for a figure with the specified name.  If it finds
% one it makes it current; if it fails it creates one with the given
% name and any prefix.
% 
% FIG(ID, ...) passes any unrecognised options to the figure.
%
% OPTIONS:
% 'new'		[off]	force new fig, appending sequence number to name
% 'prefix'	[hostname]	prefix for figure name
% See also: FIGURE.

% maneesh.
% pre-20010416: created
% 20130703: general prefix option; doc cleanup

% OPTIONS:
new = [];     % [off] force new figure, appending sequence number to name
prefix = 1;   % [hostname] prefix for figure name
figprops=optlistassign(who, varargin);

if nargin < 1
  h = figure;
  return;
end

if (~isempty(prefix) && prefix)
  if (~ischar(prefix))
    hostname=getenv('HOSTNAME');
    if(isempty(hostname))
      hostname=getenv('HOST');
    end
    if(isempty(hostname))
      [stat,hostname] = unix('uname -n');
    end
    hostname=strtok(hostname, '.');
    prefix = hostname;
  end
else
  prefix = 0;
end

if (~ischar(id))			% handle passed in
  if (ishandle(id))			% already a figure: leave it alone    
    h = figure(id);
  else					% set the new style name
    h = figure(id);
    if (prefix)
      set(h, 'name', sprintf('%s: %d', prefix, id));
    end
  end
else					% name passed in
  if prefix
    name = sprintf('%s: %s', prefix, id);
  else
    name = id;
  end

  h = findobj ('type', 'figure', 'name', name);
  if (~ isempty(h))			% already exists
    if (isempty(new))			% don't require a new figure
      set(0, 'CurrentFigure', h);	% doing it this way avoids focus issues
    else				% do require a new figure
      ii = 1;
      while 1				% look for a new name
	ii = ii + 1;
	name = sprintf('%s: %s %d', hostname, id, ii);
	h = findobj('type', 'figure', 'name', name);
	if (isempty(h))
	  h = figure('name', sprintf('%s: %s %d', hostname, id, ii));
	  break;
	end
      end
    end
  else
    h = figure('name', name);
  end
end

set(h, 'numbertitle', 'off', figprops{:});
