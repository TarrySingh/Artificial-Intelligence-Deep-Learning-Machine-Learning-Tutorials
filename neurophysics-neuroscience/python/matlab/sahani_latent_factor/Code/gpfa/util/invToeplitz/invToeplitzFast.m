%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2009
%
% invToeplitzFast()
%
% This function is simply a wrapper for the C-MEX 
% function invToeplitzFastZohar.mexa64 (or .mexglx, etc), 
% which is just a compiled version of invToeplitzFastZohar.c,
% which should also be in this folder.  Please see
% that code for details.  
% 
% This algorithm inverts a positive definite (symmetric)
% Toeplitz matrix in O(n^2) time, which is considerably better
% than the O(n^3) that inv() offers. This follows the Trench
% algorithm implementation of Zohar 1969.  See that paper for
% all explanation, as the C code is just an implementation of
% the algorithm of p599 of that paper.
%
% This function also computes the log determinant, which is 
% calculated essentially for free as part of the calculation of 
% the inverse. This is often useful in applications when one really
% needs to represent the inverse of a Toeplitz matrix. 
%
% This function should be called from within invToeplitz.m, 
% which adds a try block so that it can default to a native 
% MATLAB inversion (either inv() or a vectorized version of 
% the Trench algorithm, depending on the size of the matrix)
% should the MEX interface not work.  This will happen, for example,
% if you move to a new architecture and do not compile for .mexmaci
% or similar (see mexext('all') and help mex for some info on this).
%
% Inputs:
%    T    the positive definite (symmetric) Toeplitz matrix, which
%         does NOT need to be scaled to be 1 on the main diagonal.
%
% Outputs:
%    Ti   the inverse of T
%    ld   the log determinant of T, NOT Ti.
%
% 
% NOTE: This code is used to speed up the Toeplitz inversion as much
% as possible.  Accordingly, no error checking is done.  The onus is
% on the caller (which should be invToeplitz.m) to pass the correct args.
%
% NOTE: cf. invTopelitzFastGolub.c, which uses the algorithm in Golub and 
% Van Loan.  This newer version was written because the Zohar version 
% also computes the log determinant for free, which is essential in the 
% application for which this algorithm was coded.
%
% NOTE: Whenever possible, do not actually invert a matrix.  This code is 
% written just in case you really need to do so.  Otherwise, for example
% if you just want to solve inv(T)*x for some vector x, you are better off
% using a fast inversion method, like PCG with fast matrix multiplication,
% which could be something like an FFT method for the Toeplitz matrix.  To
% learn about this, see Cunningham, Sahani, Shenoy (2008), ICML, "Fast Gaussian
% process methods for point process intensity estimation."
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Ti,ld] = invToeplitzFast(T)

  [Ti,ld] = invToeplitzFastZohar(T);

  