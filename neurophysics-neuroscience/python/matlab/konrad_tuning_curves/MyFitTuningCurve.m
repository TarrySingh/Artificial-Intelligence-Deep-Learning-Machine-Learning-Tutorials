%%load data
load stevensonV2
%% Remove all times where speeds are very slow
isGood=find(handVel(1,:).^2+handVel(2,:).^2>.015)
handVel=handVel(1:2,isGood);
handPos=handPos(1:2,isGood);
spikes=spikes(:,isGood);
time=time(isGood);
angle=atan2(handVel(1,:),handVel(2,:));

%% Plot Raw Data - PASCAL? %%
nNeuron=193%193
clf
hold on
plot(angle,spikes(nNeuron,:)+0.2*randn(size(spikes(nNeuron,:))),'r.')

%% Make a simple tuning curve
angles=-pi:pi/8:pi;
for i=1:length(angles)-1
    angIndices=find(and(angle>angles(i),angle<=angles(i+1)));
    nSpikes(i)=mean(spikes(nNeuron,angIndices));
end
plot(angles(1:end-1)+pi/16,nSpikes)

%% PART I: KONRAD
%% bootstrap error bars
angles=-pi:pi/8:pi;
for k=1:1000
    inds=1+floor(rand(size(angle))*length(angle));
    for i=1:length(angles)-1
        angIndices=inds(and(angle(inds)>angles(i),angle(inds)<=angles(i+1)));
        nS(i,k)=mean(spikes(nNeuron,angIndices));
    end
end
nSS=sort(nS')
U=nSS(25,:);
L=nSS(975,:);
M=mean(nS')
errorbar(angles(1:end-1)+pi/16,M,M-L,U-M)
%advanced exercise: do this for all neurons. Do they actually have cosine
%tuning ad indicated by the research?

%% PART II: KONRAD
%% fit arbitrary functions
%fit a model
[bestParas,fvalCosExp(i)]=fminsearch(@whichFunction, YourInitialGuess,[],spikes(nNeuron,:),angle);

%Now plot it. 
%Here you need the function that you are actually fitting to see how the
%fit relates to the spikes
% plot(-pi:pi/80:pi, ...)))