/*=================================================================
 *
 * invToeplitzFastZohar.c completes the inversion of a symmetric
 * positive definite Toeplitz matrix.  This
 * function is a subroutine of the invToeplitz.m, which follows the 
 * algorithm of Zohar 1969.  This is the algorithm of W.F. Trench.
 *
 * The calling syntax is:
 *
 *		[ Ti , logdetT ] = invToeplitzFastZohar( T )
 *
 * T is the positive definite symmetric Toeplitz matrix of dimension n+1
 * (following the convention of p599 in the Zohar paper)
 * Ti is the inverse of T, same as inv(T) in MATLAB.  
 * logdetT is the log determinant of T, NOT of Ti.
 *
 * NOTE: This code is used to speed up the Toeplitz inversion as much
 * as possible.  Accordingly, no error checking is done.  The onus is
 * on the caller (which should be invToeplitz.m) to pass the correct args.
 * 
 * NOTE: cf. invTopelitzFastGolub.c, which uses the algorithm in Golub and 
 * Van Loan.  This newer version was written because the Zohar version 
 * also computes the log determinant for free, which is essential in the 
 * application for which this algorithm was coded.
 *
 * John P Cunningham
 * 2009
 *
 *=================================================================*/

#include <math.h>
#include "mex.h"

/* Input Arguments */

#define	T_IN	prhs[0]

/* Output Arguments */

#define	Ti_OUT	plhs[0]
#define	LD_OUT	plhs[1]

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
{ 
  double *T, *Ti;
  double *logdet;
  double *r, *lambda, *ghat, *ghatold, *gamma, *g;
  unsigned int n;
  int i,j;

    
  /* Check for proper number of arguments*/
  if (nrhs != 1)
    {
      mexErrMsgTxt("1 input arg required.");
    }
  else if (nlhs > 2) 
    {
      mexErrMsgTxt("2 output args required."); 
    }
  

  /* Check the dimensions of the arguments */
  /* This expects T to be an n+1 by n+1 matrix */
  n = mxGetN(T_IN)-1;
 

  /* Create a matrix for the return argument */ 
  Ti_OUT = mxCreateDoubleMatrix(n+1, n+1, mxREAL); 
  LD_OUT = mxCreateDoubleScalar(0);
  /* LD_OUT is a pointer to a 1x1 mxArray double initialized to 0 */
    
  /* Assign pointers to the various parameters */ 
  Ti = mxGetPr(Ti_OUT);
  logdet = mxGetPr(LD_OUT);
  T = mxGetPr(T_IN);


  r = (double*) mxMalloc(n*sizeof(double));
  if (r == NULL)
    {
      mexErrMsgTxt("Inadequate space on heap for r.");        
    }
  lambda = (double*) mxMalloc(n*sizeof(double));
  if (lambda == NULL)
    {
      mexErrMsgTxt("Inadequate space on heap for lambda.");        
    }
  ghat = (double*) mxMalloc(n*sizeof(double));
  if (ghat == NULL)
    {
      mexErrMsgTxt("Inadequate space on heap for ghat.");        
    }
  ghatold = (double*) mxMalloc(n*sizeof(double));
  if (ghatold == NULL)
    {
      mexErrMsgTxt("Inadequate space on heap for ghatold.");        
    }
  gamma = (double*) mxMalloc(n*sizeof(double));
  if (gamma == NULL)
    {
      mexErrMsgTxt("Inadequate space on heap for gamma.");        
    }
  g = (double*) mxMalloc(n*sizeof(double));
  if (g == NULL)
    {
      mexErrMsgTxt("Inadequate space on heap for g.");        
    }
  
  /* define r, the normalized row from T(1,2) to T(1,n+1) */
  for (i = 0; i < n ; i++)
    {
      r[i] = T[ 0*(n+1) + (i+1) ]/T[ 0*(n+1) + 0];
    }


  /* Initialize the algorithm */
  lambda[0] = 1 - pow(r[0],2);
  ghat[0] = -r[0];

  /* Recursion to build g and lambda */
  for (i=0 ; i < n-1 ; i++)
    {
      /* calculate gamma */
      gamma[i] = -r[i+1];
      /* ghat, g_i, etc are i+1 elts long */ 
      for (j=0 ; j < i+1 ; j++)
	{
	  gamma[i] -= r[j]*ghat[j];
	}
      /* calculate ghat */
      
      /* first store ghatold..i+1 elts long */
      for (j=0 ; j < i+1 ; j++)
	{
	  ghatold[j] = ghat[j];
	}

      ghat[0] = gamma[i]/lambda[i];
      for (j=1 ; j < i+2 ; j++)
	{
	  ghat[j] = ghatold[j-1] + gamma[i]/lambda[i]*ghatold[i+1-j];
	}
      /* calculate lambda */
      lambda[i+1] = lambda[i] - pow(gamma[i],2)/lambda[i];
    }
  /* assign g for convenience */
  for (i=0 ; i < n; i++)
    {
      g[ i ] = ghat[n-1-i];
    }

  /* There are n lambdas, n g values, and n-1 gammas */
  /* Note that n is not the dimension of the matrix, but one less. */
  /* This corresponds with Zohar notation. */


  /* Evaluate the matrix Ti (B_{n+1} in Zohar) */
  /* NOTE ON MEX MATRIX INDEXING */
  /* indexing is always (colidx * colLen + rowidx) */
  
  /* Assign first upper left element */
  Ti[ 0*(n+1) + 0 ] = 1/lambda[n-1];
  /* Assign the first row and column*/
  for (i = 1; i < n+1; i++)
    {
      /* first row */ 
      Ti[ 0*(n+1) + i] = g[ i-1 ]/lambda[ n-1 ];
      /* first column */ 
      Ti[ i*(n+1) + 0] = g[ i-1 ]/lambda[ n-1 ];
    }
  for (i = 1; i < n ; i++)
    {
      /* last row */
      Ti[ n*(n+1) + i ] = g[ n-i-1 ]/lambda[ n-1 ];
      /* last column */
      Ti[ i*(n+1) + n ] = g[ n-i-1 ]/lambda[ n-1 ];

    }
  Ti[ n*(n+1) + n ] = Ti[ 0*(n+1) + 0];


  /* Fill in the interior of Ti_OUT */
  for (i = 0; i < n/2 ; i++)
    {
      for (j = i; j < n-i-1; j++)
	{
	  /* calculate the value using p599 of Zohar */
	  Ti[ (j+1)*(n+1) + (i+1) ] = (Ti[ j*(n+1) + i ] + (1/lambda[n-1])*(g[i]*g[j] - g[n-1-i]*g[n-1-j]));
	  /* use symmetry */
	  Ti[ (i+1)*(n+1) + (j+1) ] = Ti[ ((j+1)*(n+1)) + (i+1) ]; 
	  /* use persymmetry */
	  /* recall there are n+1 elements, so 0<->n, 1<->n-1, and so on */
	  Ti[ (n-(i+1))*(n+1) + (n-(j+1)) ] = Ti[ ((j+1)*(n+1)) + (i+1) ];
	  Ti[ (n-(j+1))*(n+1) + (n-(i+1)) ] = Ti[ ((j+1)*(n+1)) + (i+1) ];
	}
    }


  /* normalize the entire matrix by T(1,1) so it is the properly scaled inverse */
  for (i = 0; i < n+1 ; i++)
    {
      for (j = 0; j < n+1 ; j++)
	{
	  Ti[ j*(n+1) + i ] = Ti[ j*(n+1) + i ]/T[ 0*(n+1) + 0];
	}
    }

  /* Calculate the log determinant for free (essentially) */
  logdet[0] = 0;
  for (i=0 ; i < n; i++)
    {
      logdet[0] += log(lambda[i]);
    }
  /* renormalize based on T(1,1) */
  logdet[0] += (n+1)*log(T[ 0*(n+1) + 0 ]);


  /* free allocated arrays */
  mxFree(g);
  mxFree(gamma);
  mxFree(ghatold);
  mxFree(ghat);
  mxFree(lambda);
  mxFree(r);

  return;
}







