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
%hint: you can sample with replacement using
inds=1+floor(rand(size(angle))*length(angle));
%another hint. Use matlab sort function
%last hint. matlab errorbar wants means, upper range (not value), lower
%range (not value) as parameters
