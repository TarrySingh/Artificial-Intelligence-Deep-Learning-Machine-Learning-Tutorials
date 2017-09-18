/*=================================================================
 * John P Cunningham
 * 2009
 * 
 * The calling syntax is:
 *
 *		[  ] = makePautoSumFast( precomp , seq )
 *
 * This function adds PautoSUM to the precomp structure.  PautoSUM is the 
 * posterior covariance of the latent variables, given data and a model in
 * the GPFA algorithm (see GPFA references noted elsewhere).  Importantly,
 * the precomp structure is modified in place, that is, this function
 * is pass-by-reference.  Though nothing unusual in C, the MATLAB user
 * must be careful, as call by reference is not typical in MATLAB.  So,
 * calling makePautoSumFast(precomp, seq) will change the precomp struct
 * without any return argument.
 * 
 * The following group post may be helpful: 
 * http://www.mathworks.com/matlabcentral/newsreader/view_thread/164276
 * We explicitly want to avoid doing a mxDuplicateArray, as that would be
 * hugely wasteful, since we are trying to optimize this code. 
 * 
 * This part of the code is offloaded to MEX because a very costly for
 * loop is required for the posterior matrix computation.  To see a perhaps
 * more readable version of what this computation is doing, see the caller
 * of this function: makePrecomp.m.  That function calls this MEX in a try
 * catch block, and it will default to a native MATLAB version if the MEX
 * is unsuccessful.  That native MATLAB version is doing precisely the same
 * calculation, albeit much slower (we find roughly a 10x speedup using this
 * C/MEX code).  This speedup is gained because the main computation has 
 * MATLAB-inefficient for loops which can be significantly parallelized/pipelined
 * and done with less overhead in C.
 *
 *=================================================================*/

#include <math.h>
#include "mex.h"

/* Input Arguments */

#define	precomp_IN	prhs[0]
#define	seq_IN	        prhs[1]

/* Output Arguments */

/* None, precomp is altered in place */


void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
{ 
  double *tmp, *xsm, *VsmGP, *nList; 
  mxArray *Tu;
  int xDim, T, numTrials, trialLens;
  int i,j,n,k,l;
  
  /* Check for proper number of arguments*/
  if (nrhs != 2)
    {
      mexErrMsgTxt("2 inputs args required.");
    }
  else if (nlhs > 0) 
    {
      mexErrMsgTxt("0 output args only...precomp is modified in place"); 
    }
  
  /* get parameters */
  xDim = mxGetNumberOfElements(precomp_IN);
  
  
  /* loop once for each state dimension */
  for (i = 0; i < xDim; i++)
    {
      /* get the appropriate precomp substruct */
      Tu = mxGetField(precomp_IN,i,"Tu");
      /* pull trialLens from here */
      trialLens = mxGetNumberOfElements(Tu);
      
      /* loop once for each unique trial length */
      for (j = 0; j < trialLens ; j++)
	{
	  /* get the appropriate Tu struct from precomp */
	  /* the length of this trial */
	  T = (int) *mxGetPr(mxGetField(Tu,j,"T"));
	  numTrials = (int) *mxGetPr(mxGetField(Tu,j,"numTrials"));
	  /* get the appropriate list of trials */
	  nList = mxGetPr(mxGetField(Tu, j , "nList"));
	  
	  /* We should be able to get that field from the struct, just like xsm and VsmGP */
	  /* and then mess with it in place. */
	  tmp = mxGetPr(mxGetField(Tu,j,"PautoSUM"));

	  /* loop once for each trial */
	  for (n = 0; n < numTrials ; n++)
	    {
	      /* get the appropriate sequence */
	      xsm = mxGetPr(mxGetField(seq_IN , (int) nList[n]-1 , "xsm"));
	      VsmGP = mxGetPr(mxGetField(seq_IN , (int) nList[n]-1 , "VsmGP"));
	      /* now do the matrix multiplication and add to tmp */
	      for (k = 0; k < T; k++)
		{
		  for (l = 0; l < T; l++)
		    {
		      /* this is the main multiplication */
		      tmp[l*T + k] += xsm[k*xDim + i]*xsm[l*xDim + i];
		      tmp[l*T + k] += VsmGP[i*T*T + l*T  + k];
		    }
		}
	    }
	}
    }
  return;
}
