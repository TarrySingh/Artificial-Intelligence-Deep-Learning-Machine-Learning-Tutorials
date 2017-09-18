%Write a header file.
%This should include
%This program performs exploratory data analysis, specifically
%raster plots and PSTHs
%It presupposes that the data file is in the same directory.
%It produces suitable figures
%Who did this (your name) and how can I reach them? (Email?)
%When? Version history V1: 11/11/2016 (Rasters and PSTHs)

%Dataset used: Eric Lindberg and Marc Slutzky
%196 neurons simultaneously, 180 trials, center-out task
%Monkey makes motions center-out with a robot arm
%Stevenson et al. 2011

%% 0 Init (Most logical errors result from lingering things
%in memory that you forgot about. So you want to start with a blank
%slate. Kind of

%Obligatory - blank slate
clear all %Clears memory from all variables
close all %Close all existing figures
clc %Clear screen
%Set all constants here. So if you want to
%change something about the analysis
%you do it in one place. 
numCond = 8; %This represent knowledge about the experiment
%There are 8 conditions. We repopulate the blankslated
%program with useful information.
%Camel case: Allows meaningful variable names that are easy to read
neuron = 193; %Which neuron are we looking at?

%Facultative initializations 
conditionIndices = cell(numCond,1); %We will put trial indices of a given condition here
rasters = cell(numCond,1); %We preallocate the cell that will contain all rasters
plotLocations = [1 2 3 6 9 8 7 4]; %This presupposed knowledge
%about which condition is which, i.e. if one is upper left
%Monkey reaches clockwise. We will use this to place
%the raster plots
scrSz = get(0,'ScreenSize'); %Get screen size - figures are children of the screen
binWidth = 50; %ms

%% 1 Loader - or transducer

load('StevensonV4.mat') %This already is a MATLAB

%% 2 Pruner - or thalamus
%We need to identify "bad data" before processing further

temp = mean(spikes); %Mean spiking across neurons over time
figure
plot(temp)
%We visually identified a problem
%We need to eliminate all trials after no more new
%data is coming in. They are valid
sum(temp(676790:end)) %This is the problematic bin
temp2 = find(startBins<676790)
lastBin = startBins(temp2(end)+1) %Start bin of the first invalid trial
startBins(:,temp2(end)+1:end) = []; %Eliminate all the ones after the critical one
targetNumbers(temp2(end)+1:end,:) = []%

%% 3 Data formatting - this is our V1
%Usually, we will spend most of our time here - 
%Most analyses fall right out of properly formatted data
%But this is already a pretty pre-formatted dataset,
%so we won't do much

%We need to find all trials of a given condition
%In other words, we need to parse the stream of targets

for ii = 1:numCond %Going through all conditions
    conditionIndices{ii,1} = find(targetNumbers==ii); 
end
%We need to use a cell structure to capture the
%start times of all trials of a certain condition
%because there is an uneven number of trials
%per condition

edges = [startBins lastBin] %This will be something we use
%to parse the data. This contains the temporal onsets


%% 4a Exploratory analysis - raster plot

%General philosophy: Do it for one trial
%Before doing it for all but with the same logic 
cond = 1; %We start with condition 1
temp = spikes(neuron,edges(conditionIndices{cond}(1)) ...
    :edges(conditionIndices{cond}(1)+1))
%... Ellipses allow to continue command on next line

%Develop logic for one, then scale for all
%We will now repurpose this logic to make *all*
%raster plots
for cond = 1:numCond %Do all conditions. Loop over all conditions
%To do this efficiently, we need a nested cell - 
%first level for all conditions, then one for all trials per condition
%Because the number of trials won't be the same for each
%condition and each trial will also not necessarily have the same length
rasters{cond,1} = cell(length(conditionIndices{cond,1}),1)

for trials = 1:length(conditionIndices{cond,1})
    rasters{cond,1}{trials,1} = ...
        spikes(neuron,edges(conditionIndices{cond}(trials)): ...
        edges(conditionIndices{cond}(trials)+1));
        end
end

%% 5a Plotting the raster
figure
spikeBins = find(temp==1); %Find the bins with spikes
%The line command draws lines, one per column
%So we need to make a matrix that does this all in one go
%So we don't need a for loop
line([spikeBins;spikeBins],[zeros(1,length(spikeBins)); ...
    ones(1,length(spikeBins))],'color','k')

figure
for ii = 1:numCond %Go through all conditions
    subplot(3,3,plotLocations(ii)) %Locations from preallocation
    for jj = 1:length(rasters{ii,1}) %Go through all trials of a given condition
        spikeBins = find(rasters{ii,1}{jj,1} ==1); %Find spike Bins of a given trial
        
        line([spikeBins;spikeBins],[ones(1,length(spikeBins)) ...
            .*jj-1; ones(1,length(spikeBins)).*jj],'color','k')
        ylim([0 length(rasters{ii,1})]) %Make the right ylim
    end
    ii %Give some feedback where we are, as this might take a while
end
set(gcf,'color','w') %Make background white
set(gcf,'Position',scrSz) %Make it big. Facultative
shg %show graph

%% 4b Exploratory analyis - PSTH

PSTHs = cell(numCond,2); %Preallocate this as a cell with two columns
allBins = []; %Preallocate empty

%Do it for one
for cond = 1:numCond %Go through all conditions
    for ii = 1:length(rasters{cond,1}) %Go through all rasters
        trialLength(ii) = length((rasters{cond,1}{ii,1})); %Get length of each trial
    end
    sharedTrialLength = min(trialLength); %PSTH only makes sense if you have data from all trials of a given condition
    flatSpikeIndices = []; %This contains something we can bin. Put it out of the loop and it will accumulate
    
    sharedRaster = zeros(length(rasters{cond,1}),sharedTrialLength); %Preallocate shared raster with zeros of right number of trials and length
    for ii = 1:length(rasters{cond,1}) %Go through all trials of a given condition 
        sharedRaster(ii,:) = rasters{cond,1}{ii,1}(1:sharedTrialLength); %Replace the zeros with spikes of shared length
        flatSpikeIndices = cat(2,flatSpikeIndices,find(sharedRaster(ii,:)==1)); %Make a large list of spike indices per condition
    end
    edges2 = 0:binWidth:sharedTrialLength; %Make a new edges vector to bin
    
    PSTHs{cond,1} = histc(flatSpikeIndices,edges2)./length(rasters{cond,1}).*(1000/binWidth); %Sum per bin divided by number of trials converted to ips
    PSTHs{cond,2} = edges2; %Capture the time base
    allBins = cat(2,allBins,PSTHs{cond,1}); %Get the firing rates for all Bins 
end




%% 5b Plotting the PSTH
figure %Open a new figure
for ii = 1:numCond %Go through all conditions
    subplot(3,3,plotLocations(ii)) %Open subplots in the right location
    plot(PSTHs{ii,2}(1:end-1),PSTHs{ii,1}(1:end-1)) %Don't plot last bin
    xlabel('time in ms') %Don't forget x label
    ylabel('FR (ips)') %Don't forget y label
    ylim([0 max(allBins)]) %Put them all on the right y-axis (first don't do this, to compare)
end
set(gcf,'color','w') %Same as before, make background white
set(gcf,'Position',scrSz) %Same as before, make figure big
shg %Show it
