function [S,Q] = genlouvain(B,limit,verbose,randord,randmove)
%GENLOUVAIN  Louvain-like community detection, specified quality function.
%   Version 2.0 (July 2014)
%
%   [S,Q] = GENLOUVAIN(B) with matrix B implements a Louvain-like greedy
%   community detection method using the modularity/quality matrix B that
%   encodes the quality function Q, defined by summing over all elements
%   B(i,j) such that nodes i and j are placed in the same community.
%   Following Blondel et al. 2008, the algorithm proceeds in two phases
%   repeated iteratively: quality is optimized by moving one node at a time
%   until no such moves improve quality; the communities found to that
%   point are then aggregated to build a new network where each node
%   represents a community.  The output vector S encodes the obtained
%   community assignments, with S(i) identifying the community to which
%   node i has been assigned.  The output Q gives the quality of the
%   resulting partition of the network.
%
%   [S,Q] = GENLOUVAIN(B) with function handle B such that B(i) returns
%   the ith column of the modularity/quality matrix uses this function
%   handle (to reduce the memory footprint for large networks) until the
%   number of groups is less than 10000 and then builds the B matrix
%   corresponding to the new aggregated network in subsequent passes.  Use
%   [S,Q] = GENLOUVAIN(B,limit) to change this default=10000 limit.
%
%   [S,Q] = GENLOUVAIN(B,limit,0) suppresses displayed text output.
%
%   [S,Q] = GENLOUVAIN(B,limit,verbose,0) forces index-ordered (cf.
%   randperm-ordered) consideration of nodes, for deterministic results.
%
%   [S,Q]=GENLOUVAIN(B,limit,verbose,randord,1) enables additional 
%   randomization to obtain a broader sample of the quality function landscape 
%   and mitigates some undesirable behavior for "multislice" modularity with 
%   ordinal coupling. Without 'randmove' enabled, the algorithm exhibits an 
%   abrupt change in behavior when the strength of the interslice coupling 
%   approaches the maximum value of the intraslice modularity matrices. With 
%   'randmove' enabled, the algorithm moves the node under consideration to a 
%   community chosen uniformly at random from all moves that increase the 
%   quality function, instead of choosing the move that maximally increases the 
%   quality function.
%
%   Example (using adjacency matrix A)
%         k = full(sum(A));
%         twom = sum(k); 
%         B = @(v) A(:,v) - k'*k(v)/twom;
%         [S,Q] = genlouvain(B); 
%         Q = Q/twom;
%     finds community assignments for the undirected network encoded by the
%     symmetric adjacency matrix A.  For small networks, one may obtain
%     reasonably efficient results even more simply by handling the full
%     modularity/quality matrix
%         B = A - k'*k/twom;
%     instead of the function handle.  Intended use also includes the
%     "multislice" network quality function of Mucha et al. 2010, where B
%     encodes the interactions as an equivalent matrix (see examples posted
%     online at http://netwiki.amath.unc.edu/GenLouvain).
%
%   Notes:
%     The matrix represented by B must be both symmetric and square.  This
%     condition is not checked thoroughly if B is a function handle, but is
%     essential to the proper use of this routine.
%
%     Under default options, this routine can return different results from
%     run to run because it considers nodes in pseudorandom (randperm)
%     order.  Because of the potentially large number of nearly-optimal
%     partitions (Good et al. 2010), one is encouraged to investigate
%     results of repeated applications of this code (and, if possible, of
%     other computational heuristics).  To force deterministic behavior,
%     ordering nodes by their index, pass zero as the fourth input:
%     GENLOUVAIN(B,limit,verbose,0).
%
%     This algorithm is only "Louvain-like" in the sense that the two
%     phases are used iteratively in the same manner as in the Louvain
%     algorithm (Blondel et al. 2008).  Because it operates on a general
%     quality/modularity matrix B, it does not include any analytical
%     formulas for quickly identifying the change in modularity from a
%     proposed move nor any improved efficiency obtained by their use.  If
%     your problem uses one of the well-used null models included in other
%     codes, those codes should be much faster for your task.
%
%     Past versions had a problem where accumulated subtraction error might
%     lead to an infinite loop with each pass oscillating between two or
%     more partitions yet incorrectly identifying increases in quality.  We
%     believe this problem has been corrected by the relative change checks
%     in lines 178 and 269.  If you encounter a similar problem, notify
%     Peter Mucha (<a href="mailto:mucha@unc.edu">mucha@unc.edu</a>).
%
%     The output Q provides the sum over the appropriate elements of B
%     without any rescaling.  As such, we have rescaled Q in the example
%     above by 2m = sum(k) so that Q <= 1.
%
%     The '~' for ignoring function returns (used for "max" below) are not
%     supported prior to R2009b.  Replace (e.g. 'dummy') for pre-2009b.
%
%     By using this code, the user implicitly acknowledges that the authors
%     accept no liability associated with that use.  (What are you doing
%     with it anyway that might cause there to be a potential liability?!?)
%
%   References:
%     Blondel, Vincent D., Jean-Loup Guillaume, Renaud Lambiotte, and
%     Etienne Lefebvre, "Fast unfolding of communities in large networks,"
%     Journal of Statistical Mechanics: Theory and Experiment, P10008
%     (2008).
%
%     Fortunato, Santo, "Community detection in graphs," Physics Reports
%     486, 75-174 (2010).
%
%     Mucha, Peter J., Thomas Richardson, Kevin Macon, Mason A. Porter, and
%     Jukka-Pekka Onnela. "Community Structure in Time-Dependent,
%     Multiscale, and Multiplex Networks," Science 328, 876-878 (2010).
%
%     Porter, M. A., J. P. Onnela, and P. J. Mucha, "Communities in
%     networks," Notices of the American Mathematical Society 56, 1082-1097
%     & 1164-1166 (2009).
%
%   Acknowledgments:
%     A special thank you to Stephen Reid, whose greedy.m code was the
%     original version that has over time developed into the present code, 
%     and Marya Bazzi for noticing the problematic behavior of genlouvain for
%     ordinal interslice coupling and contributing code that developed into the 
%     'randmove' option.
%     Thank you also to Dani Bassett, Jesse Blocher, Mason Porter and Simi
%     Wang for inspiring improvements to the code.
%
%   Citation: If you use this code, please cite as
%        Inderjit S. Jutla, Lucas G. S. Jeub, and Peter J. Mucha,
%        "A generalized Louvain method for community detection implemented
%        in MATLAB," http://netwiki.amath.unc.edu/GenLouvain (2011-2014).

%set default for maximum size of modularity matrix
if nargin<2||isempty(limit)
    limit = 10000;
end

%set level of reported/displayed text output
if nargin<3||isempty(verbose)
    verbose = 1;
end
if verbose
    mydisp = @(s) disp(s);
else
    mydisp = @(s) disp('');
end

%set randperm- v. index-ordered
if nargin<4||isempty(randord)
    randord = 1;
end
if randord
    myord = @(n) randperm(n);
else
    myord = @(n) 1:n;
end

%set move function (maximal (original Louvain) or random improvement)
if nargin<5||isempty(randmove)
    randmove=false;
end
if randmove
    movefunction='moverand';
else
    movefunction='move';
end

%initialise variables and do symmetry check
if isa(B,'function_handle')
    n=length(B(1));
    S=(1:n)';
    M=B;
    it(:,1)=M(1);
    ii=find(it(2:end)>0,3)+1;
    ii=[1,ii'];
    for i=2:length(ii),
        it(:,i)=M(ii(i));
    end
    it=it(ii,:);
    if norm(it-it')>2*eps
        error('Function handle does not correspond to a symmetric matrix')
    end
else
    n = length(B);
    S = (1:n)';
    
    if nnz(B-B'),
        B=(B+B')/2; disp('WARNING: Forced symmetric B matrix')
    end
    M=B;
end

dtot=0; %keeps track of total change in modularity

%Run using function handle, if provided
while (isa(M,'function_handle')) %loop around each "pass" (in language of Blondel et al) with B function handle
      
    y = unique(S);  %unique also puts elements in ascending order
    
    Sb=S;  
    
   
    clocktime=clock;
    mydisp(['Merging ',num2str(length(y)),' communities  ',num2str(clocktime(4:6))]);
    
    dstep=1;	%keeps track of change in modularity in pass
    yb=[];
    while (~isequal(yb,y))&&(dstep/dtot>2*eps) %This is the loop around Blondel et al's "first phase"
        yb = y;
        dstep=0;
        group_handler('assign',y);
        for i=myord(length(M(1)))
            di=group_handler(movefunction,i,M(i));
            dstep=dstep+di;
        end
        
        dtot=dtot+dstep;
        y=group_handler('return');
        mydisp([num2str(length(unique(y))),' change: ',num2str(dstep),...
            ' total: ',num2str(dtot),' relative: ',num2str(dstep/dtot)]);
    
    end
    
    %group_handler implements tidyconfig
    for i=1:length(y)
        S(S==i)=y(i);
    end
    
    %calculate modularity and return if converged
    if isequal(Sb,S)
        Q=0;
        P=sparse(y,1:length(y),1);
        for i=1:length(M(1))
            Q=Q+(P*M(i))'*P(:,i);
        end
        return
    end
    
    %check wether #groups < limit 
    t = length(unique(S));
    if (t>limit)
       metanetwork_reduce('assign',S); %inputs group information to metanetwork_reduce
       M=@(i) metanetwork_i(B,t,i); %use function handle if #groups>limit 
    else
        metanetwork_reduce('assign',S);
        J = zeros(t);   %convert to matrix if #groups small enough
        for c=1:t
            J(:,c)=metanetwork_i(B,t,c);
        end
        B = J;
        M=B;
    end
    
end
    

S2 = (1:length(B))';
Sb = [];
while ~isequal(Sb,S2) %loop around each "pass" (in language of Blondel et al) with B matrix
    
    y = unique(S2);  %unique also puts elements in ascending order
    Sb = S2;

    clocktime=clock;
    mydisp(['Merging ',num2str(length(y)),' communities  ',num2str(clocktime(4:6))]);

    yb = [];
    
    dstep=1;
    
    while (~isequal(yb,y)) && (dstep/dtot>2*eps) %This is the loop around Blondel et al's "first phase"
        
        % mydisp([num2str(length(unique(y))),' ',num2str(Q)])
        yb = y;
        dstep=0;
        group_handler('assign',y);
        for i = myord(length(M))
            di=group_handler(movefunction,i,M(:,i));
            dstep=dstep+di;
        end
        dtot=dtot+dstep;
        y=group_handler('return');
    end
    
    for i = 1:length(y)
        S(S==i) = y(i);
        S2(S2==i) = y(i);
    end
    
    if isequal(Sb,S2)
    	P=sparse(y,1:length(y),1);
    	Q=sum(sum((P*M).*P));
    	return
    end
   
    M = metanetwork(B,S2);    
end

%-----%
function M = metanetwork(J,S)
%Computes new aggregated network (communities --> nodes)
    PP = sparse(1:length(S),S,1);
    M = PP'*J*PP;
    M=full(M);


%-----%
function Mi = metanetwork_i(J,t,i) 
%ith column of metanetwork (used to create function handle)
%J is a function handle
  Mi=zeros(t,1);
  ind=metanetwork_reduce('nodes',i);
  for j=ind
      Jj=J(j);
      P=metanetwork_reduce('reduce',Jj);
      Mi=Mi+P;
  end
   


    
        
    

