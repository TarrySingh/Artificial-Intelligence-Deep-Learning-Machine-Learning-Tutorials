%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Population Latent Methods - SfN short course tutorial script 
%   Maneesh Sahani
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This file provides a skeleton notebook for the tutorial.  You can
% run each part between '%%' marks individually (Control-Enter in the
% MATLAB editor).

% The instructions on what to fill in appear within the file.

%% Add helper functions to path
addpath('util');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate a simulated data set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data.  

% Set parameters:

% These are fixed
nNeuron = 100;                          % number of neurons recorded
nLatent = 4;                            % number of generative latents

% These can be changed
nTime = 50;                             % time points per trial
nTrial = 100;                           % number of trials
aligned = 1;                            % alignment across trials

% X - [nNeuron x nTime x nTime]  contains per-trial binned counts
% Ytrue - [nLatent x nTime x nTrial] contains generative latent trajactories
% Ctrue - [nNeuron x nLatent] 'loadings' of firing rate on latents

[X,Ytrue,~,Ctrue] = GenerateData('nTime', nTime, ...
                         'nTrial', nTrial, ...
                         'aligned', aligned);


                   
%% Concatenate trials for PCA/FA
% Xcat - [nNeuron, nTime*nTrial]
Xcat  = reshape(X, nNeuron, []);

%% show the first few trials

% the plotting code below is not stock MATLAB.
% It creates a named figure; selects the top subplot and then
%   creates multiple 'nested' axes within it.

fig 'single-trial data'; nestplot(5, 1, 1);
nestable on; 
  % plottile generates nested plots, each showing a 'slice' of the array
  plottile(Ytrue(:,:,1:10), 'slice', [2,3]); 
  title (nestgroup, 'true');            % nestgroup returns the container
  axis(nestgroup('peers'), 'tight');     % nestgroup('peers') returns the nested axes 
nestable off;


%% Find mean PSTHs by averaging over trials
psth = mean(X,3);                       % [nNeuron x nTime]

% also the mean over trials and time
xbar = mean(psth, 2);


%% Show mean latents

fig 'mean activity'; nestplot(5, 1, 1);
nestable on; 
  plottile(mean(Ytrue, 3), 'slice', 2);               
  title (nestgroup, 'true');           
  axis(nestgroup('peers'), 'tight');   
nestable off;
drawnow;

%% Show true loadings

fig 'loadings'; nestplot(5, 2, {1, 1.5}, 'yspacing', 0.2);
plot(Ctrue);
title('true');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run PCA on psth to give 
%  PC vectors collected into a matrix VPCApsth
%  a diagonal matrix or vector of variances - LPCApsth 

% TASK >> 
VPCApsth = 
LPCApsth = 

% Plot the variances in descending order.  
%   What dimension is suggested?
%   How does this change with Ntrial?
fig 'PCA variances';
nestplot(2, 1, 1); 
plot(diag(LPCApsth), '-*');        % CHANGE if your LPCApsth is a vector


% Compute the PCA latents (into YPCApsth) by truncating.  Try dimensions
% corresponding to nLatent (the true value defined above), your own guess
% if different, and nLatent + 2.

% TASK >>
VPCApsth = % [truncate]
YPCApsth = 

% plot them:
fig 'mean activity'; nestplot(5, 1, 2);
nestable on; 
  plottile(YPCApsth, 'slice', 2);   
  title (nestgroup, 'PCA'); 
  axis(nonzeros(nestgroup('peers')), 'tight');
nestable off;
drawnow;

% Plot the truncated loadings
fig 'loadings'; nestplot(5, 2, {2,1});
plot(VPCApsth); 
title('PCA mean');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Now repeat using Xcat to get VPCAtrial, LPCAtrial etc.

% TASK >> 
VPCAtrial = 
LPCAtrial = 

% Plot the variances in descending order.  
fig 'PCA variances';
nestplot(2, 1, 2); 
plot(diag(LPCAtrial), '-*');

% Compute the PCA latents (into YPCAtrial) by truncating.  
% Rearrange into a 3-D array (like X)

% TASK >>
YPCAtrial= 

% Plot latents for some trials, and the mean
fig 'single-trial data'; nestplot(5, 1, 2);
nestable on; 
   plottile(YPCAtrial(:,:,1:10), 'slice', 2:3); 
   set(nestgroup('peers'), 'nextplot', 'add');
   plottile(mean(YPCAtrial, 3), 'slice', 2, 'color', 'k', 'linewidth', 2); 
   title(nestgroup, 'PCA');
   axis(nestgroup('peers'), 'tight');
nestable off;

%% Plot the truncated loadings
fig 'loadings'; nestplot(5, 2, {2,2});
plot(VPCAtrial); 
title('PCA single trials');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you have time: go back and run everything again with aligned 
%% set to 0 in GenerateData.
% TASK >>




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FACTOR ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is functional code.  Your assignment is to vary parameters
% and interpret the results.

% TASK >>
% Try these:
%  experiment with the number of latents
%  try rotating to the varimax (or other) basis -- see ROTATEFACTORS()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run factor analysis on psth
[VFApsth,UFApsth,~,YFApsth] = fa_em(psth, nLatent);

% When you rotate (see above) do it here, and uncomment the line below
% TASK >> 
% YFApsth = fa_infer(psth, VFApsth, UFApsth); % Recalculate latents after rotation

% plot the FA latents
fig 'mean activity'; nestplot(5, 1, 3);
nestable on; 
  plottile(YFApsth, 'slice', 2);   
  title (nestgroup, 'Factor Analysis'); 
  axis(nestgroup('peers'), 'tight');
nestable off;
drawnow;

% Plot the loadings
fig 'loadings'; nestplot(5, 2, {3,1});
plot(VFApsth); 
title('FA mean');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Repeat using single-trial data in Xcat

[VFAtrial, UFAtrial,~, YFAtrial] = fa_em(Xcat, nLatent);
YFAtrial = reshape(YFAtrial, [], nTime, nTrial);

% rotate (and uncomment re-inference)
% TASK >> 
% YFAtrial = fa_infer(Xcat, VFAtrial, UFAtrial);
% YFAtrial = reshape(YFAtrial, [], nTime, nTrial);


% Plot latents for some trials, and the mean
fig 'single-trial data'; nestplot(5, 1, 3);
nestable on; 
   plottile(YFAtrial(:,:,1:10), 'slice', 2:3); 
   set(nestgroup('peers'), 'nextplot', 'add');
   plottile(mean(YFAtrial, 3), 'slice', 2, 'color', 'k', 'linewidth', 2); 
   title(nestgroup, 'FA');
   axis(nestgroup('peers'), 'tight');
nestable off;

%% Plot the loadings
fig 'loadings'; nestplot(5, 2, {3,2});
plot(VFAtrial); 
title('FA single trials');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GPFA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Again, this is complete, so you should experiment with:
% TASK >>
%   Number of latents
%   Plotting xorth vs. xsm (see below)
%   aligned = 0
%   Cross-validation: look at gpfa/example.m


% Add GPFA code base to path
% Written by Byron Yu and John Cunningham, to accompany 
% Yu et al. 2009, J. Neurophysiol.
addpath(genpath('gpfa'));

% Construct a data structure in the format that the GPFA code
% expects.
dat = struct('trialId', num2cell([1:nTrial]'), ...
             'spikes',  squeeze(num2cell(X,1:2))...
             ); 

% Arguments controlling timescales. GPFA code expects spiketrains
% binned at 1ms; we have counts at a nominal 20ms.  So don't bin further.
dataArgs = {'binWidth', 1, 'segLength', nTime, 'startTau', 5};

% Results will be saved in mat_results/runXXX/, where XXX is runIdx.  
% GPFA will not rerun if saved results exist, so remove (or change
% runIdx) to force it.
runIdx = aligned;

% Learn GPFA model and latent trajectories
GPFAresult = neuralTraj(runIdx, dat, 'method', 'gpfa', dataArgs{:},...
                        'xDim', nLatent);

% Orthonormalize neural trajectories to ranked variance projection
[estParams, GPFAtrials] = postprocess(GPFAresult);
% NOTE: See pp.621-622 of Yu et al., J Neurophysiol, 2009.

% TASK >>
% Try looking at both of:
% xsm - smoothed trajectories, separated by time constant.
% xorth - orthonormalised trajectories
YGPFAtrial = cat(3, GPFAtrials(:).xsm);
VGPFAtrial = estParams.C;
% YGPFAtrial = cat(3, GPFAtrials(:).xorth);
% VGPFAtrial = estParams.Corth;


% plot the GPFA latent for some trials, and the mean
fig 'single-trial data'; nestplot(5, 1, 4);
nestable on; 
   plottile(YGPFAtrial(:,:,1:10), 'slice', 2:3); 
   set(nestgroup('peers'), 'nextplot', 'add');
   plottile(mean(YGPFAtrial, 3), 'slice', 2, 'color', 'k', 'linewidth', 2); 
   title(nestgroup, 'GPFA');
   axis(nestgroup('peers'), 'tight');
nestable off;

%% Plot the truncated loadings
fig 'loadings'; nestplot(5, 2, {4,2});
plot(VGPFAtrial); 
title('GPFA single trials');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LGSSM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TASK >>
% Again, done for you.  But you get the idea ... experiment!!


%% Run LGSSM on psth 
[SSMpsth, ~] = ssm_em(bsxfun(@minus, psth, xbar), 'latentdim', nLatent);
YSSMpsth = ssm_kalman(bsxfun(@minus, psth, xbar), SSMpsth, 'smooth');
VSSMpsth = SSMpsth.output;

% plot the LGSSM latents
fig 'mean activity'; nestplot(5, 1, 5);
nestable on; 
  plottile(YSSMpsth, 'slice', 2); 
  title (nestgroup, 'SSM'); 
  axis(nestgroup('peers'), 'tight');
nestable off;
drawnow;

% Plot the loadings
fig 'loadings'; nestplot(5, 2, {5,1});
plot(VSSMpsth); 
title('SSM mean');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Repeat using single-trial data in X (trial structure matters)

[SSMtrial, ~] = ssm_em(bsxfun(@minus, X, xbar), 'latentdim', nLatent);
for nn = 1:nTrial
  YSSMtrial(:,:,nn) = ssm_kalman(bsxfun(@minus, X(:,:,nn), xbar), SSMtrial, 'smooth');
end
VSSMtrial = SSMtrial.output;

% Plot latents for some trials, and the mean
fig 'single-trial data'; nestplot(5, 1, 5);
nestable on; 
   plottile(YSSMtrial(:,:,1:10), 'slice', 2:3); 
   set(nestgroup('peers'), 'nextplot', 'add');
   plottile(mean(YSSMtrial, 3), 'slice', 2, 'color', 'k', 'linewidth', 2); 
   title(nestgroup, 'SSM');
   axis(nestgroup('peers'), 'tight');
nestable off;

%% Plot the loadings
fig 'loadings'; nestplot(5, 2, {5,2});
plot(VSSMtrial); 
title('SSM single trials');

