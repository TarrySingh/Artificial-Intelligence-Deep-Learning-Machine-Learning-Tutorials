function remain = optlistassign (opts, varargin)
% OPTLISTASSIGN - assign optional arguments.
%
%   REM = OPTLISTASSIGN(OPTLIST, 'VAR1', VAL1, 'VAR2', VAL2, ...)
%   assigns, in the caller's workspace, the values VAL1,VAL2,... to
%   the variables that appear in the cell array OPTLIST and that match
%   the strings 'VAR1','VAR2',... .  Any VAR-VAL pairs that do not
%   match a variable in OPTLIST are returned in the cell array REM.
%   The VAR-VAL pairs can also be passed to OPTLISTASSIGN in a cell
%   array: REM = OPTLISTASSIGN(OPTLIST, {'VAR1', VAL1, ...}); If
%   OPTLISTASSIGN encounters a struct in place of a 'VAR' string anywhere
%   in the list it expands the struct into additional arguments,
%   treating the fieldnames as variable names.
%
%   By default OPTLISTASSIGN matches option names using the strmatch
%   defaults: matches are case sensitive, but a (unique) prefix is
%   sufficient.  If a 'VAR' string is a prefix for more than one
%   option in OPTLIST, and does not match any of them exactly, no
%   assignment occurs and the VAR-VAL pair is returned in REM.  
%
%   This behaviour can be modified by preceding OPTLIST with one or
%   both of the following flags:
%      'ignorecase' implies case-insensitive matches.
%      'exact'      implies exact string matches.
%   Both together imply case-insensitive, but otherwise exact, matches.
%
%   OPTLISTASSIGN useful for processing optional arguments to a function.
%   Thus in a function which starts:
%		function foo(x,y,varargin)
%		z = 0;
%		optlistassign({'z'}, varargin{:});
%   the variable z can be given a non-default value by calling the
%   function thus: foo(x,y,'z',4);  When used in this way, a list
%   of currently defined variables can easily be obtained using
%   WHO.  Thus if we define:
%		function foo(x,y,varargin)
%		opt1 = 1;
%               opt2 = 2;
%		rem = optlistassign('ignorecase', who, varargin);
%   and call foo(x, y, 'OPT1', 10, 'opt', 20); the variable opt1
%   will have the value 10, the variable opt2 will have the
%   (default) value 2 and the list rem will have the value {'opt',
%   20}. 
% 
%   See also OPTLISTWARN, WHO.

ignorecase = 0;
exact = 0;

% check for flags at the beginning
while (~iscell(opts))
  switch(lower(opts))
   case 'ignorecase',
    ignorecase = 1;
   case 'exact',
    exact = 1;
   otherwise,
    error(['unrecognized flag :', opts]);
  end
  
  opts = varargin{1};
  varargin(1) = [];
end

% if passed cell array instead of list, deal
if length(varargin) == 1 & iscell(varargin{1})
  varargin = varargin{1};
end

% look for arg structs in place of vars
i = 1;
while i <= length(varargin)
  if (isstruct(varargin{i}))
    ssargs = struct2args(varargin{i});
    varargin = {varargin{1:i-1}, ...
                ssargs{:}, ...
                varargin{i+1:end}};
    i = i+length(ssargs);
  else
    i = i+2;
  end
end

% list has been processed: should now only have {'var', val} pairs.
if rem(length(varargin),2)~=0,
   error('Optional arguments and values must come in pairs')
end     

done = zeros(1, length(varargin));

origopts = opts;
if ignorecase
  opts = lower(opts);
end

for i = 1:2:length(varargin)

  opt = varargin{i};
  if ignorecase
    opt = lower(opt);
  end
  
  % look for matches
  
  if exact
    match = strmatch(opt, opts, 'exact');
  else
    match = strmatch(opt, opts);
  end
  
  % if more than one matched, try for an exact match ... if this
  % fails we'll ignore this option.

  if (length(match) > 1)
    match = strmatch(opt, opts, 'exact');
  end

  % if we found a unique match, assign in the corresponding value,
  % using the *original* option name
  
  if length(match) == 1
    assignin('caller', origopts{match}, varargin{i+1});
    done(i:i+1) = 1;
  end
end

varargin(find(done)) = [];
remain = varargin;

function args = struct2args(ss)

fields = fieldnames(ss);
args = cell(1,2*length(fields));
for  ff = 1:length(fields)
  args{2*ff-1} = fields{ff};
  args{2*ff} = ss.(fields{ff});
end
