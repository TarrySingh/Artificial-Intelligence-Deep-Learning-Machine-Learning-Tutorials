function yOut = smoother(yIn, kernSD, stepSize)
%
% yOut = smoother(yIn, kernSD, stepSize)
%
% Gaussian kernel smoothing of data across time.
%
% INPUTS:
%
% yIn      - input data (yDim x T)
% kernSD   - standard deviation of Gaussian kernel, in msec
% stepSize - time between 2 consecutive datapoints in yIn, in msec
%
% OUTPUTS:
%
% yOut     - smoothed version of yIn (yDim x T)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  if (kernSD == 0) || (size(yIn, 2)==1)
    yOut = yIn;
    return
  end

  % Filter half length
  % Go 3 standard deviations out
  fltHL = ceil(3 * kernSD / stepSize);

  % Length of flt is 2*fltHL + 1
  flt = normpdf(-fltHL*stepSize : stepSize : fltHL*stepSize, 0, kernSD);

  % Ensure that filter taps sum to 1
  flt = flt / sum(flt);

  [yDim, T] = size(yIn);
  yOut      = nan(yDim, T);

  % Want to normalize by sum of filter taps actually used
  nm = conv(flt, ones(1, T));
  
  for i = 1:yDim
    ys = conv(flt, yIn(i,:)) ./ nm;
    % Cut off edges so that result of convolution is same length 
    % as original data
    yOut(i,:) = ys(fltHL+1:end-fltHL);
  end
