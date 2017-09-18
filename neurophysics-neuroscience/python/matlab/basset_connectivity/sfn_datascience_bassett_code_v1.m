%% BASSETT SFN Data Science TUTORIAL MATLAB SCRIPT

% Representing interaction patterns as graphs

% Task 1: Identify and characterize local network structure in neural data
% Task 2: Using graph-based community detection methods to identify modules in neural data 
% Task 3: Use dynamic extensions of these tools to describe how modules change over time

addpath(genpath('./'))

%% TASK 1
% Identify and characterize local network structure in neural data

% Now that we understand what a clustering coefficient is, let's calculate
% it on some real data. First, we'll open up the adjacency matrices that we
% have been given, which represent functional connectivity (estimated by a
% wavelet coherence in task-evoked BOLD in the frequency band 0.06-0.12 Hz)
% between 112 cortical and subcortical brain areas (defined by the
% Harvard-Oxford atlas).

M = load('matrices.mat');

% Let's start with the first matrx.

matrix1 = squeeze(M.matrices(:,:,1));

% Calculate the clustering coefficient of the matrix using the Brain
% Connectivity Toolbox, available online at
% https://sites.google.com/site/bctnet

C = clustering_coef_wu(matrix1);

% What is the distribution of clustering coefficients over nodes in the
% network?

figure; hist(C);

% What is the average clustering coefficient for the whole network?

C_global = mean(C);

% Do any brain regions have more or less clustering than expected? To
% answer this question, we need to define a null hypothesis. Let's decide
% that our null hypothesis is that all brain regions connect to one another
% with probability, p, a property that is characteristic of a random (or
% so-called Erdos-Renyi) graph? To test whether brain graphs show the same
% properties as graphs under the null hypothesis, we must construct a
% random matrix, and recalculate the clustering coefficient.

uppertriangle= find(triu(matrix1,1)>0);
matrixr = zeros(size(matrix1));
matrixr(uppertriangle) = matrix1(uppertriangle(randperm(numel(uppertriangle)))); 
matrixr = matrixr+matrixr';
Cr = clustering_coef_wu(matrixr);

% Let's look at the distribution of clustering coefficients in the random
% network overlad on the distribution from the real network.

figure; hist([C'; Cr']')

% Is this result consistent with our intuition from the matrices
% themselves?

figure; subplot(1,2,1); imagesc(matrix1); subplot(1,2,2); imagesc(matrixr);



%% TASK 2
% Using graph-based community detection methods to identify modules in neural data 

% Now that we understand what a network community is, and what modularity
% maximization is, let's find some network communities in real data. Again,
% we will use the Brain Connectivity Toolbox, available online at
% https://sites.google.com/site/bctnet/. The inputs to this algorithm are
% 3-fold: the adjacency matrix A, the resolution parameter gamma whose
% default value is unity, and the initialization community structure (where
% each node is in its own community).

n = numel(matrix1(:,1));
[partition1,Q1]=community_louvain(matrix1,1,randperm(n))

% Do we get the same answer every time?
[partition2,Q2]=community_louvain(matrix1,1,randperm(n))
[partition3,Q3]=community_louvain(matrix1,1,randperm(n))
[partition4,Q4]=community_louvain(matrix1,1,randperm(n))
[partition5,Q5]=community_louvain(matrix1,1,randperm(n))
[partition6,Q6]=community_louvain(matrix1,1,randperm(n))

% No, and why? Well, modularity maximization is NP-hard, and we only have
% heuristic algorithms to get us close to the answer (for a thorough
% description of the near-degeneracy of the modularity landscape). Are the
% values of the modularity index close? and are the partitions similar?

Q_values = [Q1 Q2 Q3 Q4 Q5 Q6]
partitions = [partition1 partition2 partition3 partition4 partition5 partition6];
figure; imagesc(partitions);

% Yes, the values of Q are similar, and the partitions of brain regions
% into clusters are similar as well. Are the values of Q or the partitions
% different from what you would expect in a random matrix? Let's do the
% same calculations on the variable matrixr that we made in Task 1.

[partitionr1,Qr1]=community_louvain(matrixr,1,randperm(n))
[partitionr2,Qr2]=community_louvain(matrixr,1,randperm(n))
[partitionr3,Qr3]=community_louvain(matrixr,1,randperm(n))
[partitionr4,Qr4]=community_louvain(matrixr,1,randperm(n))
[partitionr5,Qr5]=community_louvain(matrixr,1,randperm(n))
[partitionr6,Qr6]=community_louvain(matrixr,1,randperm(n))
Qr_values = [Qr1 Qr2 Qr3 Qr4 Qr5 Qr6]
partitionsr = [partitionr1 partitionr2 partitionr3 partitionr4 partitionr5 partitionr6];

% First, notice that the modularity values of the random graph are much
% lower than that observed in your real data. This indicates that your real
% data has significant modularity (which can be proven with appropriate
% statistics). Next, notice that the partitions in your real data are quite
% different than the partitions in your random data:

figure; subplot(1,2,1); imagesc(partitions); subplot(1,2,2); imagesc(partitionsr);

% Let's spot check how this maps on to the adjacency matrix representing
% the relationships between brain regions:

[indSort,CiSort] = fcn_order_partition(matrix1,partition1);
[x,y]=fcn_grid_communities(CiSort);
figure; imagesc(matrix1(indSort,indSort)); hold on; plot(x,y,'k');

% And we can compare to what we would see in the random graph like this:

[indSort,CiSort] = fcn_order_partition(matrixr,partitionr1);
[x,y]=fcn_grid_communities(CiSort);
figure; imagesc(matrixr(indSort,indSort)); hold on; plot(x,y,'k');



%% TASK 3
% Use dynamic extensions of these tools to describe how modules change over time

% Now that we know what multilayer modularity is, let's calculate it on
% some real data. In the original data file, we have 9 adjacency matrices.
% And in our fictitious world, let's assume that those 9 instances are
% ordered in time. Then, we can identify network communities as a function
% of time using this multilayer extension of the modularity maximization
% approach.

gamma = 1.2;
omega = 1;
T=length(squeeze(M.matrices(1,1,:)));
B=spalloc(n*T,n*T,n*n*T+2*n*T);
twomu=0;
for s=1:T
    k=sum(squeeze(M.matrices(:,:,s)));
    twom=sum(k);
    twomu=twomu+twom;
    indx=[1:n]+(s-1)*n;
    B(indx,indx)=squeeze(M.matrices(:,:,s))-gamma*k'*k/twom;
end
twomu=twomu+2*omega*n*(T-1);
B = B + omega*spdiags(ones(n*T,2),[-n,n],n*T,n*T);
[S,Q] = genlouvain(B);
Q = Q/twomu;
partition_multi = reshape(S,n,T);

% What does the partition of nodes into communities look like?

figure; imagesc(partition_multi)

% well, it looks like there are 4 communities, that are expressed very
% consistently over time. 

% Extra questions:

% Q1. What happens if we tune the value of omega down to 0.2? (( re-run the
% above with omega=0.2)).
% A1. Well, we see that there is more variability in the partitions from
% time step to time step. This is a key feature of the algorithm: it can be
% used to probe slow time scale features of your data (using large values
% of omega) and fast time scale features of your data (using small values
% of omega).

% Q2. What happens if we tuen the value of gamma up to 1.2? ((re-run the
% above with gamma=1.2)). 
% A2. We see that instead of 5 communities, we now get 28 communities. This
% is a second key feature of the algorithm: it can be used to probe
% community structure at small topological scales (using large values of
% gamma that partition the network into few communities). 

%% End of Bassett SFN Data Science Tutorial

