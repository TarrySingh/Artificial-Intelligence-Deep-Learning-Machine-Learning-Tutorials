
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2009
%
% invToeplitz()
%
% Invert a symmetric, real, positive definite Toeplitz matrix
% using either inv() or the Trench algorithm, which 
% uses Zohar 1969.  This is slightly different than 
% Algorithm 4.7.3 of Golub and Van Loan 
% "Matrix Computations," the 1996 version, because the Zohar
% version produces the log determinant essentially for free, which
% is often useful in cases where one actually needs the full matrix inverse.
%
% If Zohar or the MATLAB implementation of the Trench algorithm is called,
% the inversion is done in O(n^2) time, which is considerably better
% than the O(n^3) that inv() offers. This follows the Trench
% algorithm implementation of Zohar 1969.  See that paper for
% all explanation, as the C code is just an implementation of
% the algorithm of p599 of that paper.  See Algorithm 4.7.3 of Golub and Van Loan
% for an explanation of the MATLAB version of this algorithm, which is slower
% due to for loops (slower than inv(), up to about n=300).  
% Thus, if the C can be used, always use it.
%
% This function also computes the log determinant, which is 
% calculated essentially for free as part of the calculation of 
% the inverse when Zohar is used. This is often useful in applications when one really
% needs to represent the inverse of a Toeplitz matrix. If Zohar is not used, the computation
% is reasonably costly.
%
% Usage: [Ti,ld] = invToeplitz(T, [runMode]);
%
% Inputs: 
%     T         the positive definite symmetric Toeplitz matrix to be inverted
%     runMode   OPTIONAL: to force this inversion to happen using a particular method.  This
%               will NOT roll over to a method (by design), so it will fail if something is amuck.
% 
% Outputs: 
%     Ti        the inverted matrix, also symmetric (and persymmetric), NOT TOEPLITZ
%
% NOTE: We bother with this method because we 
% need this particular matrix inversion to be 
% as fast as possible.  Thus, no error checking
% is done here as that would add needless computation.
% Instead, the onus is on the caller to make sure 
% the matrix is toeplitz and symmetric and real.
%
% NOTE: Algorithm 4.7.3 in the Golub book has a number of typos
% Do not use Alg 4.7.3, use the equations on p197-199.
%
% Run time tests (testInvToeplitz) suggest that Zohar is the fastest method across 
% the board.  (2-3 orders of magnitude at sizes from 200-1000).  Trench is 13/4n^2 flops, 
% whereas inv() is probably 2/3n^3.  However, the overhead for the MEX calls, etc., 
% may make the crossover point very MATLAB dependent.  
%
% NOTE: The interested numerical linear algebra geek can dig into all this stuff by
% going into the directory 'tests' and running that testInvToeplitz to find the crossover 
% point him/herself. Some tinkering required.  This will run seven inversion methods: inv(), invToeplitzFastGolub (MEX),
% invToeplitzFastZohar (MEX), invPerSymm (2x2 exploitation of persymmetry), invToeplitz
% (Golub MATLAB non-vectorized...almost always the worst), and invToeplitz (Golub MATLAB vectorized).
% Use figure 1 from the results.  This version of invToeplitz has been pared down to 
% be more user friendly, so it only includes Zohar, inv(), and vectorized MATLAB Trench/Golub.
%
% MEX NOTE: This function will call a MEX routine.
% A try block is included to default back to the best non-MEX solver (either inv() 
% or a vectorized Trench/Golub version in native MATLAB), so hopefully the user will
% not experience failures.  We have also included the compiled .mex in a number of
% different architectures (hopefully all).  However, the user should really compile
% this code on his/her own machine to ensure proper use of MEX.  That can be done
% from the MATLAB prompt with "mex invToeplitzFast.c". This C code does nothing 
% fancy, so if you can run MEX at all, this should work.  See also 'help mex' and 
% 'mexext('all')'.
% 
% MEX NOTE 2: The try-catch block is to hopefully fail MEX gracefully.  If the mex
% code is not compiled or in the wrong architecture, this will default to the next
% best MATLAB implementation, which, depending on n, is inv() or vectorized Trench.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ti,ld] = invToeplitz( T , runMode )

  % dimension of the matrix
  n = size(T,1);
  % if runMode is specified, just run with that (for testing, etc.)
  % otherwise, do automated determination of runMode
  if nargin<2 || isempty(runMode)
    % then do the automated determination of runMode, based on runtime tests
    % always try Zohar first.
    tryMode = 0;
    % if MEX fails, use inv() or vectorized Trench in MATLAB
    % The n=150 is a rough estimate based on tests on 64 and 32 bit linux systems
    if n < 150
      % inv() is the best failure option
      catchMode = -1;
    else
      % n is >=150, so vectorized Trench is the best failure option
      catchMode = 1;
    end
  else
    % if specified, force it
    tryMode = runMode;
    catchMode = runMode;
  end
  
  % Invert T with the specified method
  try 
    [Ti,ld] = invToeplitzMode( T , tryMode);
  catch
    [Ti,ld] = invToeplitzMode( T , catchMode);
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION FOR TRY-CATCH BLOCK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ti,ld] = invToeplitzMode( T , runMode )

  % dimension of the matrix
  n = size(T,1);
  % Invert T to produce Ti, the inverse, by the specified method
  switch (runMode)
    
   case -1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % just call MATLAB inv()
    Ti = inv(T);
    % and calculate the log determinant
    ld = logdet(T);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
   case 0
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % here is the fast C-MEX implementation of this inversion, using Zohar
    % invToeplitzFastZohar.c should be mex-compiled with a command from the MATLAB
    % prompt of "mex invToeplitzFastZohar.c"  This should just work.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % call the MEX wrapper
    [Ti,ld] = invToeplitzFast(T); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
   case 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % a faster vectorized version of Trench that wastes no computations
    % This is faster than 2 but considerably less fast than 0.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % set up the Trench parameters
    [r,gam,v] = trenchSetup(T);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fill out the borders of Ti
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % the first element
    Ti(1,1) = gam;
    % the first row
    Ti(1,2:n) = v(n-1:-1:1)';
    % use symmetry and persymmetry to fill out border of matrix
    % the first column 
    Ti(2:n,1) = Ti(1,2:n)';
    % last column - persymmetric with 1st row
    Ti(2:n,n) = Ti(1,n-1:-1:1)';
    % last row - persymmetric with 1st column
    Ti(n,2:n-1) = Ti(n-1:-1:2,1)';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fill out the interior of Ti
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = (n-1) : -1 : (floor((n-1)/2) + 1);
      % 4.7.5
      Ti(i,i:-1:(n-i+1)) = Ti(n-(i:-1:(n-i+1)),n-i) + (1/gam)*((v(i)*v(i:-1:(n-i+1)))' - v(n-i)*v(n-(i:-1:(n-i+1)))');
      % symmetry
      Ti((n-i+1):(i-1),i) = Ti(i,(n-i+1):(i-1));
      % persymmetry
      Ti((n-i+1),(i-1):-1:(n-i+1)) = Ti((n-i+2):(i),i)'; % note 1 fewer element assigned
      % symmetry
      Ti((n-i+1-1:i-1),(n-i+1)) = Ti((n-i+1),(n-i+1-1:i-1)); % note 1 fewer elements again
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % renormalize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ti = Ti/T(1,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calculate the log determinant
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ld = logdet(T);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION FOR TRENCH SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [r,gam,v] = trenchSetup(T);

  % dimension of the matrix
  n = size(T,1);
  % Initial setup for Trench algorithm (from Golub book)
  % normalize the matrix to r0 = 1 w.l.o.g.
  r = [T(1,2:n)]/T(1,1); 
  % nicely, MATLAB has an implementation of Algorithm 4.7.1 to solve Tn-1*y = (-r1,...,rn-1)';
  % it does not quite seem to work as MATLAB says it does, but the following reconciles it with 4.7.1.
  % y = T(1:n-1,1:n-1)\(-T(1,2:end)'); % this considerably slows the implementation.
  y = levinson([1 r])'; % this is a reasonable piece of the computation in the fastest algorithm (2n^2 of 13/4n^2 flops)
  y = y(2:end);
  % calculate the quantity gamma
  % gam = 1/(1+r(1:n-1)*y(n-1:-1:1)); (This is a typo from the book)
  % this calculates the right answer
  gam = 1/(1+r(1:n-1)*y(1:n-1));
  % now calculate v
  v(1:n-1) = gam*y(n-1:-1:1); % corresponds to v = gam*E*y, top of p198
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%