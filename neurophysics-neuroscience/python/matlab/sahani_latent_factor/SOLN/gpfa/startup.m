addpath core_gpfa
addpath core_twostage
addpath plotting
addpath util
addpath util/precomp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following code checks for the relevant MEX files (such as .mexa64
% or .mexglx, depending on the machine architecture), and it creates the
% mex file if it can not find the right one.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Toeplitz Inversion
path(path,'util/invToeplitz');
% Create the mex file if necessary.
if ~exist(sprintf('util/invToeplitz/invToeplitzFastZohar.%s',mexext),'file')
  try
    eval(sprintf('mex -outdir util/invToeplitz util/invToeplitz/invToeplitzFastZohar.c'));
    fprintf('NOTE: the relevant invToeplitz mex files were not found.  They have been created.\n');
  catch
    fprintf('NOTE: the relevant invToeplitz mex files were not found, and your machine failed to create them.\n');
    fprintf('      This usually means that you do not have the proper C/MEX compiler setup.\n');
    fprintf('      The code will still run identically, albeit slower (perhaps considerably).\n');
    fprintf('      Please read the README file, section Notes on the Use of C/MEX.\n');
  end
end
  
% Posterior Covariance Precomputation  
path(path,'util/precomp');
% Create the mex file if necessary.
if ~exist(sprintf('util/precomp/makePautoSumFast.%s',mexext),'file')
  try
    eval(sprintf('mex -outdir util/precomp util/precomp/makePautoSumFast.c'));
    fprintf('NOTE: the relevant precomp mex files were not found.  They have been created.\n');
  catch
    fprintf('NOTE: the relevant precomp mex files were not found, and your machine failed to create them.\n');
    fprintf('      This usually means that you do not have the proper C/MEX compiler setup.\n');
    fprintf('      The code will still run identically, albeit slower (perhaps considerably).\n');
    fprintf('      Please read the README file, section Notes on the Use of C/MEX.\n');
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
