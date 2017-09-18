function h = nestable(varargin)
% nestable - control nestability of axes: H=nestable([AX],['on']|['off',P])
%
% NESTABLE returns a handle to the current axes, unless they are not
% nestable, in which case it returns the handle of the corresponding
% container.
%
% NESTABLE('on') makes the current axes nestable, and returns a handle
% to them.  
%
% NESTABLE('off') makes the first *container* at or above the current axes 
% non-nestable, and returns a handle to the container above it in
% the nestplot tree.  The container must have a parent -- that is,
% it must be a nested plot.
%
% NESTABLE('off', container) makes the first *container* at or above
% the current axes non-nestable, sets its container to the
% specified handle, and returns a handle to the new container. 
%
% NESTABLE(H, ...) performs any of the above operations on the axes H.
%
% See also: NESTPLOT.  

% maneesh.
% pre-20020324: created
% 20130702: doc cleanup
% 20130704: switched data store from userdata to appdata
% 20140424: initialise empty children array in NestData
% 20140501: bugfix -- nestable(h) now steps up as far as needed.
%           nestable(...,'on') adds a resize listener

narg = nargin;

if (narg > 0 & ~ischar(varargin{1}))
  chkaxes = varargin{1};
  varargin = {varargin{2:end}};
  narg = narg - 1;
else
  chkaxes = gca;
end

switch (narg)
  case 0,

   h = chkaxes;
   ndat = getappdata(h, 'NestData');
   
   while isfield(ndat, 'nestable') && ~ndat.nestable
     h = ndat.container;
     ndat = getappdata(h, 'NestData');
   end

   
  otherwise,

    switch varargin{1}
      case 'on',
	ndat = getappdata(chkaxes, 'NestData');
	if (isfield(ndat, 'nestable'))
	  ndat.nestable = 1;
	else
	  ndat = struct('nestable', 1, 'container', 0);
	end
        if ~isfield(ndat, 'children')
          ndat.children = [];
        end
        c_hand = handle(chkaxes);
        ndat.resizelistener = ...
            addlistener(c_hand, 'Position', 'PostSet',...
                            @(x,y) nestresize(chkaxes));

	setappdata (chkaxes, 'NestData', ndat);
	h = chkaxes;
	
      case 'off',
       chkaxes = nestable(chkaxes);
       ndat = getappdata(chkaxes, 'NestData');
	if isfield(ndat, 'nestable')
	  ndat.nestable = 0;
	  if (narg < 2)
	    if ~ishandle(ndat.container)
	      error ('No container specified')
	    end
	  else
	    ndat.container   = varargin{2};
	  end
	else
	  if (narg < 2) 
            error ('No container specified')
          end
          ndat = struct('nestable', 0, 'container', varargin{2});
	end
	setappdata(chkaxes, 'NestData', ndat);
	h = ndat.container;
      
      otherwise,
	error('usage: nestable|nestable on|nestable(''off'',P)|nestable(H,...)');
    end    

end    
