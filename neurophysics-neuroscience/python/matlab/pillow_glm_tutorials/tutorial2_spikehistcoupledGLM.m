% tutorial2_spikehistcoupledGLM.m
%
% This is an interactive tutorial designed to walk you through the steps of
% fitting an autoregressive Poisson GLM (i.e., a spiking GLM with
% spike-history) and a multivariate autoregressive Poisson GLM (i.e., a
% GLM with spike-history AND coupling between neurons).
%
% (Data from Uzzell & Chichilnisky 2004; see README.txt file in data
% directory for details). 
%
% Last updated: Nov 10, 2016 (JW Pillow)


% Instructions: Execute each section below separately using cmd-enter.
% For detailed suggestions on how to interact with this tutorial, see
% header material in tutorial1_PoissonGLM.m

%% ====  1. Load the raw data ============

datdir = 'data_RGCs/';  % directory where stimulus lives
load([datdir, 'Stim']);     % stimulus (temporal binary white noise)
load([datdir,'stimtimes']); % stim frame times in seconds (if desired)
load([datdir, 'SpTimes']);  % load spike times (in units of stimulus frames)
ncells = length(SpTimes);  % number of neurons (4 for this dataset).
% Neurons #1-2 are OFF, #3-4 are ON.

% Compute some basic statistics on the stimulus
dtStim = (stimtimes(2)-stimtimes(1)); % time bin size for stimulus (s)
nT = size(Stim,1); % number of time bins in stimulus

% See tutorial 1 for some code to visualize the raw data!

%% ==== 2. Bin the spike trains =========================
%
% For now we will assume we want to use the same time bin size as the time
% bins used for the stimulus. Later, though, we'll wish to vary this.
tbins = (.5:nT)*dtStim; % time bin centers for spike train binnning
sps = zeros(nT,ncells);
for jj = 1:ncells
    sps(:,jj) = hist(SpTimes{jj},tbins)';  % binned spike train
end

% Let's just visualize the spike-train auto and cross-correlations
% (Comment out this part if desired!)
clf;
nlags = 30; % number of time-lags to use 
for ii = 1:ncells
    for jj = ii:ncells
        % Compute cross-correlation of neuron i with neuron j
        xc = xcorr(sps(:,ii),sps(:,jj),nlags,'unbiased');

        % remove center-bin correlation for auto-correlations (for ease of viz)
        if ii==jj, xc(nlags+1) = 0;
        end
        
        % Make plot
        subplot(ncells,ncells,(ii-1)*ncells+jj);
        plot((-nlags:nlags)*dtStim,xc,'.-','markersize',20); 
        axis tight; drawnow;
        title(sprintf('cells (%d,%d)',ii,jj)); axis tight;
    end
end
xlabel('time shift (s)');

%% ==== 3. Build design matrix: single-neuron GLM with spike-history =========

% Pick the cell to focus on (for now).
cellnum = 3;  % 1-2: OFF, 3-4: ON

% Set the number of time bins of stimulus to use for predicting spikes
ntfilt = 25;  % Try varying this, to see how performance changes!
% Set number of time bins of auto-regressive spike-history to use
nthist = 20;

% Build stimulus design matrix (using 'hankel');
paddedStim = [zeros(ntfilt-1,1); Stim]; % pad early bins of stimulus with zero
Xstim = hankel(paddedStim(1:end-ntfilt+1), Stim(end-ntfilt+1:end));

% Build spike-history design matrix
paddedSps = [zeros(nthist,1); sps(1:end-1,cellnum)];
% SUPER important: note that this doesn't include the spike count for the
% bin we're predicting? The spike train is shifted by one bin (back in
% time) relative to the stimulus design matrix
Xsp = hankel(paddedSps(1:end-nthist+1), paddedSps(end-nthist+1:end));

% Combine these into a single design matrix
Xdsgn = [Xstim,Xsp];

% Let's visualize the design matrix just to see what it looks like
subplot(1,10,1:9); 
imagesc(1:(ntfilt+nthist), 1:50, Xdsgn(1:50,:));
xlabel('regressor');
ylabel('time bin of response');
title('design matrix (including stim and spike history)');
subplot(1,10,10); 
imagesc(sps(1:50,cellnum));
set(gca,'yticklabel', []); 
title('spike count');

% The left part of the design matrix has the stimulus values, the right
% part has the spike-history values.  The image on the right is the spike
% count to be predicted.  Note that the spike-history portion of the design
% matrix had better be shifted so that we aren't allowed to use the spike
% count on this time bin to predict itself!

%% === 4. fit single-neuron GLM with spike-history ==================

% First fit GLM with no spike-history
fprintf('Now fitting basic Poisson GLM...\n');
pGLMwts0 = glmfit(Xstim,sps(:,cellnum),'poisson'); % assumes 'log' link and 'constant'='on'.
pGLMconst0 = pGLMwts0(1);
pGLMfilt0 = pGLMwts0(2:end);

% Then fit GLM with spike history (now use Xdsgn design matrix instead of Xstim)
fprintf('Now fitting Poisson GLM with spike-history...\n');
pGLMwts1 = glmfit(Xdsgn,sps(:,cellnum),'poisson');
pGLMconst1 = pGLMwts1(1);
pGLMfilt1 = pGLMwts1(2:1+ntfilt);
pGLMhistfilt1 = pGLMwts1(ntfilt+2:end);

%%  Make plots comparing filters
ttk = (-ntfilt+1:0)*dtStim; % time bins for stim filter
tth = (-nthist:-1)*dtStim; % time bins for spike-history filter

clf; subplot(221); % Plot stim filters
h = plot(ttk,ttk*0,'k--',ttk,pGLMfilt0, 'o-',ttk,pGLMfilt1,'o-','linewidth',2);
legend(h(2:3), 'GLM', 'sphist-GLM','location','northwest');axis tight;
title('stimulus filters'); ylabel('weight');
xlabel('time before spike (s)');

subplot(222); % Plot spike history filter
colr = get(h(3),'color');
h = plot(tth,tth*0,'k--',tth,pGLMhistfilt1, 'o-');
set(h(2), 'color', colr, 'linewidth', 2); 
title('spike history filter'); 
xlabel('time before spike (s)');
ylabel('weight'); axis tight;

%% Plot predicted rate out of the two models

% Compute predicted spike rate on training data
ratepred0 = exp(pGLMconst0 + Xstim*pGLMfilt0);
ratepred1 = exp(pGLMconst1 + Xdsgn*pGLMwts1(2:end));

% Make plot
iiplot = 1:60; ttplot = iiplot*dtStim;
subplot(212);
stem(ttplot,sps(iiplot,cellnum), 'k'); hold on;
plot(ttplot,ratepred0(iiplot),ttplot,ratepred1(iiplot), 'linewidth', 2);
hold off;  axis tight;
legend('spikes', 'GLM', 'hist-GLM');
xlabel('time (s)');
title('spikes and rate predictions');
ylabel('spike count / bin');

%% === 5. fit coupled GLM for multiple-neuron responses ==================

% First step: build design matrix containing spike history for all neurons

Xspall = zeros(nT,nthist,ncells); % allocate space
% Loop over neurons to build design matrix, exactly as above
for jj = 1:ncells
    paddedSps = [zeros(nthist,1); sps(1:end-1,jj)];
    Xspall(:,:,jj) = hankel(paddedSps(1:end-nthist+1),paddedSps(end-nthist+1:end));
end

% Reshape it to be a single matrix
Xspall = reshape(Xspall,nT,[]);
Xdsgn2 = [Xstim, Xspall]; % full design matrix (with all 4 neuron spike hist)

clf; % Let's visualize 50 time bins of full design matrix
imagesc(1:1:(ntfilt+nthist*ncells), 1:50, Xdsgn2(1:50,:));
title('design matrix (stim and 4 neurons spike history)');
xlabel('regressor');
ylabel('time bin of response');

%% Fit the model (stim filter, sphist filter, coupling filters) for one neuron 

fprintf('Now fitting Poisson GLM with spike-history and coupling...\n');

pGLMwts2 = glmfit(Xdsgn2,sps(:,cellnum),'poisson');
pGLMconst2 = pGLMwts2(1);
pGLMfilt2 = pGLMwts2(2:1+ntfilt);
pGLMhistfilts2 = pGLMwts2(ntfilt+2:end);
pGLMhistfilts2 = reshape(pGLMhistfilts2,nthist,ncells);

% So far all we've done is fit incoming stimulus and coupling filters for
% one neuron.  To fit a full population model, redo the above for each cell
% (i.e., to get incoming filters for 'cellnum' = 1, 2, 3, and 4 in turn).  


%% Plot the fitted filters and rate prediction

clf; subplot(221); % Plot stim filters
h = plot(ttk,ttk*0,'k--',ttk,pGLMfilt0, 'o-',ttk,pGLMfilt1,...
    ttk,pGLMfilt2,'o-','linewidth',2); axis tight; 
legend(h(2:4), 'GLM', 'sphist-GLM','coupled-GLM', 'location','northwest');
title(['stimulus filter: cell ' num2str(cellnum)]); ylabel('weight'); 
xlabel('time before spike (s)');

subplot(222); % Plot spike history filter
colr = get(h(3),'color');
h = plot(tth,tth*0,'k--',tth,pGLMhistfilts2,'linewidth',2);
legend(h(2:end),'from 1', 'from 2', 'from 3', 'from 4', 'location', 'northwest');
title(['coupling filters: into cell ' num2str(cellnum)]); axis tight;
xlabel('time before spike (s)');
ylabel('weight');

% Compute predicted spike rate on training data
ratepred2 = exp(pGLMconst2 + Xdsgn2*pGLMwts2(2:end));

% Make plot
iiplot = 1:60; ttplot = iiplot*dtStim;
subplot(212);
stem(ttplot,sps(iiplot,cellnum), 'k'); hold on;
plot(ttplot,ratepred0(iiplot),ttplot,ratepred1(iiplot),...
    ttplot,ratepred2(iiplot), 'linewidth', 2);
hold off;  axis tight;
legend('spikes', 'GLM', 'sphist-GLM', 'coupled-GLM', 'location', 'northwest');
xlabel('time (s)');
title('spikes and rate predictions');
ylabel('spike count / bin');

%% 6. Model comparison: log-likelihoood and AIC

% Let's compute loglikelihood (single-spike information) and AIC to see how
% much we gain by adding each of these filter types in turn:

LL_stimGLM = sps(:,cellnum)'*log(ratepred0) - sum(ratepred0);
LL_histGLM = sps(:,cellnum)'*log(ratepred1) - sum(ratepred1);
LL_coupledGLM = sps(:,cellnum)'*log(ratepred2) - sum(ratepred2);

% log-likelihood for homogeneous Poisson model
nsp = sum(sps(:,cellnum));
ratepred_const = nsp/nT;  % mean number of spikes / bin
LL0 = nsp*log(ratepred_const) - nT*sum(ratepred_const);

% Report single-spike information (bits / sp)
SSinfo_stimGLM = (LL_stimGLM - LL0)/nsp/log(2);
SSinfo_histGLM = (LL_histGLM - LL0)/nsp/log(2);
SSinfo_coupledGLM = (LL_coupledGLM - LL0)/nsp/log(2);

fprintf('\n empirical single-spike information:\n ---------------------- \n');
fprintf('stim-GLM: %.2f bits/sp\n',SSinfo_stimGLM);
fprintf('hist-GLM: %.2f bits/sp\n',SSinfo_histGLM);
fprintf('coupled-GLM: %.2f bits/sp\n',SSinfo_coupledGLM);

% Compute AIC
AIC0 = -2*LL_stimGLM + 2*(1+ntfilt); 
AIC1 = -2*LL_histGLM + 2*(1+ntfilt+nthist);
AIC2 = -2*LL_coupledGLM + 2*(1+ntfilt+ncells*nthist);
AICmin = min([AIC0,AIC1,AIC2]); % the minimum of these

fprintf('\n AIC comparison (smaller is better):\n ---------------------- \n');
fprintf('stim-GLM: %.1f\n',AIC0-AICmin);
fprintf('hist-GLM: %.1f\n',AIC1-AICmin);
fprintf('coupled-GLM: %.1f\n',AIC2-AICmin);

% These are whopping differencess! Clearly coupling has a big impact in
% terms of log-likelihood, though the jump from stimulus-only to
% own-spike-history is greater than the jump from spike-history to
% full coupling.


%% Advanced exercises:
% --------------------
% 1. Write code to simulate spike trains from the fitted spike-history GLM.
% Simulate a raster of repeated responses from the stim-only GLM and
% compare to raster from the spike-history GLM

% 2. Write code to simulate the 4-neuron population-coupled GLM. There are
% now 16 spike-coupling filters (including self-coupling), since each
% neuron has 4 incoming coupling filters (its own spike history coupling
% filter plus coupling from three other neurons.  How does a raster of
% responses from this model compare to the two single-neuron models?

% 3. Compute a non-parametric estimate of the spiking nonlinearity for each
% neuron. How close does it look to exponential now that we have added
% spike history? Rerun your simulations using different non-parametric
% nonlinearity for each neuron. How much improvement do you see in terms of
% log-likelihood, AIC, or PSTH % variance accounted for (R^2) when you
% simulate repeated responses?
