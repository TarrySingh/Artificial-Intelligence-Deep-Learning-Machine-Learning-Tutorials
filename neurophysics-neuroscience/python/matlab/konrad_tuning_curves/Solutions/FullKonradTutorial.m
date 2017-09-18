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
[bestParas,fvalCosExp(i)]=fminsearch(@evaluateScoreCosExp, [.8,0.1,4],[],spikes(nNeuron,:),angle);
plot(-pi:pi/80:pi, exp(bestParas(1)+bestParas(2)*cos((-pi:pi/80:pi)-bestParas(3))))


%% PART 3 Konrad
%% now do some machine learning. Matlab does not allow us to do this with poisson distributions 
nNeuron=193; %183 141 193
%first lets have some meaningful regressors
Y=spikes(nNeuron,:)';
X=handVel(1:2,:);
X(3:4,:)=handPos(1:2,:);
X=X';

%do trivial model and linear regression first
for fold=1:100
    indsTrain=1:length(Y);
    indsTrain(find(mod(indsTrain-fold,100)==0))=[];
    indsTest=fold:100:length(Y);
    [b, bint, r, rint, stats]=regress(Y(indsTrain,1),[X(indsTrain,:), 1+0*X(indsTrain,1)]);
    pred = b'*[X(indsTest,:), 1+0*X(indsTest,1)]';
    mse(fold)=mean((Y(indsTest,1)'-pred).^2);
    mseConst(fold)=mean((Y(indsTest,1)'-mean(Y(indsTrain,1))).^2);
end
mseTotalLinearRegression=mean(mse)
mseTotalConst=mean(mseConst)

close all
rng(1945,'twister')
leaf = [12 25 50 100];
col = 'rbcm';
figure
for i=1:length(leaf)
b = TreeBagger(200,X,Y,'Method','R','OOBPred','On',...
'MinLeafSize',leaf(i));
plot(oobError(b)-mseTotalConst,col(i));
hold on
end
xlabel 'Number of Grown Trees'
ylabel 'Mean Squared Error'
legend({'12' '25' '50' '100'},'Location','NorthEast')
hold off






