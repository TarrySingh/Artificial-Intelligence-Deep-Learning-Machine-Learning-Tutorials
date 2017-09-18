function hhout = plottile(varargin)
% plottile - plot N-D array in multiple subplots: plottile([x],y,...)
% PLOTTILE(Y) plots each of the columns of the N-D array Y (N <=3) in
%  an individual subplot on the current figure.  If Y is LxMxN, the
%  function uses a grid of MxN subplots, each containing a line
%  with L points.
% PLOTTILE(X,Y) uses the vector X to set the X-axis.
% PLOTTILE(X,Y,'prop','value',...) passes the property-value pairs
%  to the underlying plots.  This can be used to set the line style
%  or color.
% H = PLOTTILE(...) returns a vector of handles to the line
%  objects.

% maneesh.
% pre-20030205: created


% OPTIONS:
slice = [1];
xy = 'False';
Plot = @plot; 
axesOpts = {};
% OPTIONS DONE

if nargin > 1 && isvector(varargin{1}) && ~ischar(varargin{2})
  xx = varargin{1};
  yy = varargin{2};
  varargin(1:2) = [];
else
  yy = varargin{1};
  xx = 1:size(yy,1);
  varargin(1) = [];
end

plargs = optlistassign(who, varargin);

if ndims(yy)-length(slice) > 2
  error ('cannot create tile arrays with more than two dimensions');
end



% bring the dimensions for each slice to the front
yy = permute(yy, [slice, setdiff(1:ndims(yy), slice)]);
ysiz = size(yy);
ssiz = ysiz(1:length(slice));
ssiz(end+1:2) = 1;

dims = ysiz;
dims(end+1:3) = 1;		% fill out the dim vector

if length(slice) > 1
  % reshape 
  yy = reshape(yy, [prod(ssiz), ysiz(length(slice)+1:end)]);
  dims = size(yy);
  dims(length(dims)+1:3) = 1;		% fill out the dim vector
end

%% create X axis
xx = 1:ssiz(1);


hh = [];

Nrows = dims(end-1);
Ncols = dims(end);

for ii = 1:Nrows
  for jj = 1:Ncols
    nestplot(Nrows, Ncols, {ii,jj});
    if ~isempty(axesOpts)
      set(gca, axesOpts{:});
    end
    hh = [hh, Plot(xx, reshape(yy(:,ii,jj), ssiz), plargs{:})];
  end      
end
    
if (nargout)    
  hhout = hh;
end
  