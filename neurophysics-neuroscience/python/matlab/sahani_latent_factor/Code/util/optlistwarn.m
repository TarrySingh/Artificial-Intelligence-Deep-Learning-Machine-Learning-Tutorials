function unassigned = optlistwarn(unassigned)
% OPTLISTWARN - warn about unassigned options.  rem=optlistwarn(assignopts(...))
%
% OPTLISTWARN('var1', val1, 'var2', val2, ...) prints a warning
% stating that 'var1', 'var2', ... are not recognised options.
% This is useful in combination with OPTLISTASSIGN.

if (length(unassigned))
  warning(['unrecognized options:', sprintf(' %s', unassigned{1:2:end})])
end
