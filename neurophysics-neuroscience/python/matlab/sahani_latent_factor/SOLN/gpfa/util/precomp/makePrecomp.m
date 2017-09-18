
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2009
%
% makePrecomp()
%
% Make the precomputation matrices specified by the GPFA algorithm.
% 
% Usage: [precomp] = makePautoSum( seq , xDim )
%
% Inputs: 
%     seq     - The sequence struct of inferred latents, etc. 
%     xDim    - the dimension of the latent space.
%     NOTE    - All inputs are named sensibly to those in learnGPparams.m.  This code probably should not
%               be called from anywhere but there.
% 
% Outputs: 
%     precomp - The precomp struct will be updated with the posterior covaraince and the other requirements.
%
% NOTE: We bother with this method because we 
% need this particular matrix sum to be
% as fast as possible.  Thus, no error checking
% is done here as that would add needless computation.
% Instead, the onus is on the caller (which should be 
% learnGPparams()) to make sure this is called correctly.
%
% NOTE: The called MEX code executes the costly for-loops more efficiently, around 10x.
%
% MEX NOTE: This function will call a MEX routine.
% A try block is included to default back to the native MATLAB version, so the user should
% not experience failures.  We have also included the compiled .mex in a number of
% different architectures (hopefully all relevant).  However, the user should really compile
% this code on his/her own machine to ensure proper use of MEX.  That can be done
% from the MATLAB prompt with "mex invToeplitzFast.c". This C code does nothing 
% fancy, so if you can run MEX at all, this should work.  See also 'help mex' and 
% 'mexext('all')'.  Finally, see the notes in the GPFA README.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function precomp = makePrecomp( seq , xDim )

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Setup
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Tall = [seq.T];
  Tmax = max(Tall);
  Tdif = repmat((1:Tmax)', 1, Tmax) - repmat(1:Tmax, Tmax, 1);
  
  % assign some helpful precomp items
  % this is computationally cheap, so we keep a few loops in MATLAB
  % for ease of readability.
  for i = 1:xDim
    precomp(i).absDif = abs(Tdif);
    precomp(i).difSq  = Tdif.^2;
    precomp(i).Tall   = Tall;
  end
  % find unique numbers of trial lengths
  Tu = unique(Tall);
  % Loop once for each state dimension (each GP)
  for i = 1:xDim
    for j = 1:length(Tu)
      T     = Tu(j);
      precomp(i).Tu(j).nList = find(Tall == T);
      precomp(i).Tu(j).T = T;
      precomp(i).Tu(j).numTrials = length(precomp(i).Tu(j).nList);
      precomp(i).Tu(j).PautoSUM  = zeros(T);
    end
  end
  
  % at this point the basic precomp is built.  The previous steps
  % should be computationally cheap.  We now try to embed the 
  % expensive computation in a MEX call, defaulting to MATLAB if 
  % this fails.  The expensive computation is filling out PautoSUM,
  % which we initialized previously as zeros.
  

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Fill out PautoSum
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  try 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run MEX for fast implementation of this matrix sum
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    makePautoSumFast(precomp, seq);
    % NOTE the use of call-by-reference.  This is unusual for 
    % MATLAB: the precomp structure is being passed to 
    % makePautoSumFast by reference, so this call will change
    % precomp even though it is not "assigned" in the output.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  catch 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run native MATLAB (MEX is not working)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop once for each state dimension (each GP)
    for i = 1:xDim
      % Loop once for each trial length (each of Tu)
      for j = 1:length(Tu)
        % Loop once for each trial (each of nList)
        for n = precomp(i).Tu(j).nList
          precomp(i).Tu(j).PautoSUM = precomp(i).Tu(j).PautoSUM +...
              seq(n).VsmGP(:,:,i) +...
              seq(n).xsm(i,:)' * seq(n).xsm(i,:);        
        end
      end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('MEX did not work in PautoSUM...defaulting to native MATLAB.\n');
  end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


