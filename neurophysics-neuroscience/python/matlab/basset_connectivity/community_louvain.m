function [M,Q]=community_louvain(W,gamma,M0,B)
%COMMUNITY_LOUVAIN     Optimal community structure
%
%   M     = community_louvain(W);
%   [M,Q] = community_louvain(W,gamma);
%   [M,Q] = community_louvain(W,gamma,M0);
%   [M,Q] = community_louvain(W,gamma,M0,'potts');
%   [M,Q] = community_louvain(W,gamma,M0,'negative_asym');
%   [M,Q] = community_louvain(W,[],[],B);
%
%   The optimal community structure is a subdivision of the network into
%   nonoverlapping groups of nodes which maximizes the number of within-
%	group edges, and minimizes the number of between-group edges.
%
%   This function is a fast and accurate multi-iterative generalization of
%   the Louvain community detection algorithm. This function subsumes and
%   improves upon,
%		modularity_louvain_und.m, modularity_finetune_und.m,
%		modularity_louvain_dir.m, modularity_finetune_dir.m,
%       modularity_louvain_und_sign.m
%	and additionally allows to optimize other objective functions (includes
%	built-in Potts-model Hamiltonian, allows for custom objective-function
%	matrices).
%
%   Inputs:
%       W,
%           directed/undirected weighted/binary connection matrix with
%           positive and possibly negative weights.
%       gamma,
%           resolution parameter (optional)
%               gamma>1,        detects smaller modules
%               0<=gamma<1,     detects larger modules
%               gamma=1,        classic modularity (default)
%       M0,     
%           initial community affiliation vector (optional)
%       B,
%           objective-function type or custom objective matrix (optional)
%           'modularity',       modularity (default)
%           'potts',            Potts-model Hamiltonian (for binary networks)
%           'negative_sym',     symmetric treatment of negative weights
%           'negative_asym',    asymmetric treatment of negative weights
%           B,                  custom objective-function matrix
%
%           Note: see Rubinov and Sporns (2011) for a discussion of
%           symmetric vs. asymmetric treatment of negative weights.
%
%   Outputs:
%       M,      
%           community affiliation vector
%       Q,  
%           optimized community-structure statistic (modularity by default)
%
%   Example:
%       % Iterative community finetuning.
%       % W is the input connection matrix.
%       n  = size(W,1);             % number of nodes
%       M  = 1:n;                   % initial community affiliations 
%       Q0 = -1; Q1 = 0;            % initialize modularity values
%       while Q1-Q0>1e-5;           % while modularity increases
%           Q0 = Q1;                % perform community detection
%           [M, Q1] = community_louvain(W, [], M);
%       end
%
%   References:
%       Blondel et al. (2008)  J. Stat. Mech. P10008.
%       Reichardt and Bornholdt (2006) Phys. Rev. E 74, 016110.
%       Ronhovde and Nussinov (2008) Phys. Rev. E 80, 016109
%       Sun et al. (2008) Europhysics Lett 86, 28004.
%       Rubinov and Sporns (2011) Neuroimage 56:2068-79.
%
%   Mika Rubinov, U Cambridge 2015-2016

%   Modification history
%   2015: Original
%   2016: Included generalization for negative weights.
%         Enforced binary network input for Potts-model Hamiltonian.
%         Streamlined code and expanded documentation.

W=double(W);                                % convert to double format
n=length(W);                                % get number of nodes
s=sum(sum(W));                              % get sum of edges

if ~exist('B','var') || isempty(B)
    type_B = 'modularity';
elseif ischar(B)
    type_B = B;
else
    type_B = 0;
    if exist('gamma','var') && ~isempty(gamma)
        warning('Value of gamma is ignored in generalized mode.')
    end
end
if ~exist('gamma','var') || isempty(gamma)
    gamma = 1;
end

if strcmp(type_B,'negative_sym') || strcmp(type_B,'negative_asym')
    W0 = W.*(W>0);                          %positive weights matrix
    s0 = sum(sum(W0));                      %weight of positive links
    B0 = W0-gamma*(sum(W0,2)*sum(W0,1))/s0; %positive modularity
    
    W1 =-W.*(W<0);                          %negative weights matrix
    s1 = sum(sum(W1));                      %weight of negative links
    if s1                                   %negative modularity
        B1 = W1-gamma*(sum(W1,2)*sum(W1,1))/s1;
    else
        B1 = 0;
    end
elseif min(min(W))<-1e-10
    err_string = [
        'The input connection matrix contains negative weights.\nSpecify ' ...
        '''negative_sym'' or ''negative_asym'' objective-function types.'];
    error(sprintf(err_string))              %#ok<SPERR>
end
if strcmp(type_B,'potts') && any(any(W ~= logical(W)))
    error('Potts-model Hamiltonian requires a binary W.')
end

if type_B
    switch type_B
        case 'modularity';      B = (W-gamma*(sum(W,2)*sum(W,1))/s)/s;
        case 'potts';           B =  W-gamma*(~W);
        case 'negative_sym';    B = B0/(s0+s1) - B1/(s0+s1);
        case 'negative_asym';   B = B0/s0      - B1/(s0+s1);
        otherwise;              error('Unknown objective function.');
    end
else                            % custom objective function matrix as input
    B = double(B);
    if ~isequal(size(W),size(B))
        error('W and B must have the same size.')
    end
end
if ~exist('M0','var') || isempty(M0)
    M0=1:n;
elseif numel(M0)~=n
    error('M0 must contain n elements.')
end

[~,~,Mb] = unique(M0);
M = Mb;

B = (B+B.')/2;                                          % symmetrize modularity matrix
Hnm=zeros(n,n);                                         % node-to-module degree
for m=1:max(Mb)                                         % loop over modules
    Hnm(:,m)=sum(B(:,Mb==m),2);
end

Q0 = -inf;
Q = sum(B(bsxfun(@eq,M0,M0.')));                        % compute modularity
first_iteration = true;
while Q-Q0>1e-10
    flag = true;                                        % flag for within-hierarchy search
    while flag;
        flag = false;
        for u=randperm(n)                               % loop over all nodes in random order
            ma = Mb(u);                                 % current module of u
            dQ = Hnm(u,:) - Hnm(u,ma) + B(u,u);
            dQ(ma) = 0;                                 % (line above) algorithm condition
            
            [max_dQ,mb] = max(dQ);                      % maximal increase in modularity and corresponding module
            if max_dQ>1e-10;                            % if maximal increase is positive
                flag = true;
                Mb(u) = mb;                             % reassign module
                
                Hnm(:,mb) = Hnm(:,mb)+B(:,u);           % change node-to-module strengths
                Hnm(:,ma) = Hnm(:,ma)-B(:,u);
            end
        end
    end
    [~,~,Mb] = unique(Mb);                              % new module assignments
    
    M0 = M;
    if first_iteration
        M=Mb;
        first_iteration=false;
    else
        for u=1:n                                       % loop through initial module assignments
            M(M0==u)=Mb(u);                             % assign new modules
        end
    end
    
    n=max(Mb);                                          % new number of modules
    B1=zeros(n);                                        % new weighted matrix
    for u=1:n
        for v=u:n
            bm=sum(sum(B(Mb==u,Mb==v)));                % pool weights of nodes in same module
            B1(u,v)=bm;
            B1(v,u)=bm;
        end
    end
    B=B1;
    
    Mb=1:n;                                             % initial module assignments
    Hnm=B;                                              % node-to-module strength
    
    Q0=Q;
    Q=trace(B);                                         % compute modularity
end
