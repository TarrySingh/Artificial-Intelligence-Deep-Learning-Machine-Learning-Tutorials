//
//  full.cpp
//  full
//
//  Created by Lucas Jeub on 24/10/2012
//
//
//  Implements thin wrapper class for full matlab matrices
//
//
//  Last modified by Lucas Jeub on 27/11/2012


#include "matlab_matrix.h"

//default constructor
full::full(): n(0), m(0), val(NULL), export_flag(0) {}


//copy constructor
full::full(const full &matrix): m(matrix.m), n(matrix.n), export_flag(0) {
	
    //allocate memory
    val=(double *) mxCalloc(m*n, sizeof(double));
	
    //copy values
	for(mwIndex i=0; i<m*n; i++){
		val[i]=matrix.val[i];
	}
}

//construct by size
full::full(mwSize m_, mwSize n_): m(m_), n(n_), export_flag(0) {
	val=(double *) mxCalloc(m*n,sizeof(double));
}


//construct from mxArray (not save, useful for input arguments that are not modified, otherwise use operator = )
full::full(const mxArray * matrix): m(mxGetM(matrix)), n(mxGetN(matrix)), export_flag(0){
    
    if (mxIsDouble(matrix)) {
        if(mxIsSparse(matrix)){
            //input is sparse
            
            //allocate memory (initialises to zero)
            val=(double *) mxCalloc(m*n,sizeof(double));
            
            //get input values
            mwSize *row=mxGetIr(matrix);
            mwSize *col=mxGetJc(matrix);
            double *pr=mxGetPr(matrix);
            
            //assign values
            for(mwIndex j=0; j<n; j++){
                for(mwIndex i=col[j]; i<col[j+1];i++){
                    val[row[i]+j*m]=pr[i];
                }
            }
        }
        else{
            //input is full
            //get input
            val= mxGetPr(matrix);
            
            export_flag=1;
        }
    }
    else{
        //wrong input
        mexErrMsgIdAndTxt("full:constructor", "mxArray must be either full or sparse matrix");
    }
}


//destructor free memory if it has not been exported  to matlab array)
full::~full(){
    
	if(!export_flag){
		mxFree(val);
	}
}


//copy assignment
full & full::operator = (const full & matrix) {
	
    //check for self assignment
	if(this!= &matrix){
        //copy size
		m=matrix.m;
		n=matrix.n;
        
        //allocate memory
		val=(double *) mxRealloc(val,m*n*sizeof(double));
        
        //copy values
		for(mwIndex i=0; i<m*n;i++){
			val[i]=matrix.val[i];
		}
	}
	
	return *this;
}


//convert from sparse
full & full::operator = (const sparse & matrix){
    
    //copy size
    m=matrix.m;
    n=matrix.n;
    
    //allocate memory
    val=(double *) mxRealloc(val, m*n*sizeof(double));
    
    //copy values
    for(mwIndex j=0; j<n; j++){
        for (mwIndex i=matrix.col[j]; i<matrix.col[j+1]; i++) {
            val[matrix.row[i]+j*m]=matrix.val[i];
        }
    }
    
    return *this;
}


//copy from mxArray
full & full::operator = (const mxArray * matrix){
	
    //get size of input
    m=mxGetM(matrix);
	n=mxGetN(matrix);
    mwSize total_size=m*n;
    if (mxIsDouble(matrix)) {
        
        if(mxIsSparse(matrix)){
            //input is sparse
        
            //allocate memory
            val=(double *) mxRealloc(val,total_size*sizeof(double));
            //initialise to 0
            for (mwIndex i=0; i<total_size; i++) {
                val[i]=0;
            }
        
            //get input values
            mwIndex *row=mxGetIr(matrix);
            mwIndex *col=mxGetJc(matrix);
            double *pr=mxGetPr(matrix);
		
            //copy input values
            for(mwIndex j=0; j<n; j++){
                for(mwIndex i=col[j]; i<col[j+1];i++){
                    get(row[i],j)=pr[i];
                }
            }
        }
        else{
            //input is full
        
            //get input values
            double * val_in=mxGetPr(matrix);
            
            //allocate memory
            val=(double *) mxRealloc(val,total_size*sizeof(double));
		
            //copy values
            
            for(mwIndex i=0; i<total_size; i++){
                val[i]=val_in[i];
            }
        }
    }
	else{
        //wrong input
        mexErrMsgIdAndTxt("full:assignment", "mxArray must be either full or sparse matrix");
	}
    
	return *this;
}


//export to full matlab mxArray (sets export flag to avoid freeing memory if used to set output argument);
void full::export_matlab(mxArray * & out){
	
    //create empty full matrix
    out = mxCreateDoubleMatrix(0,0,mxREAL);
	
    //free value pointer
	mxFree(mxGetPr(out));
    
    //set sizes
	mxSetN(out, n);
	mxSetM(out, m);
	
    //assing values
	mxSetPr(out, val);
	
    //set export flag
	export_flag=1;
}
	

//get elements by index
double & full::get(mwIndex i, mwIndex j){
    if (i<m&&j<n) {
    	return val[i+j*m];
    }
    else {
        mexErrMsgIdAndTxt("full:index", "index out of bounds");
    }
}

double full::get(mwIndex i, mwIndex j) const{
    if (i<m&&j<n) {
         return val[i+j*m];
    }
    else {
        mexErrMsgIdAndTxt("full:index:const", "index out of bounds");
    }
}

double & full::get(mwIndex i){
    if (i<m*n) {
     	return val[i];
    }
    else {
        mexErrMsgIdAndTxt("full:index:linear", "index out of bounds");
    }
}
double full::get(mwIndex i) const{
    if (i<m*n) {
       return val[i];
    }
    else {
        mexErrMsgIdAndTxt("full:index:linear:const", "index out of bounds");
    }
}

/*operations*/

/*pointwise division (note 0/0=0)*/

// by a sparse matrix
full full::operator / (const sparse &B){
	
    //check appropriate size
    if(m!=B.m || n != B.n){
		mexErrMsgIdAndTxt("Pdiv:Dimension","Matrices must have same size");
	}

    //create output matrix
    full C(m,n);

	mwIndex i,k;
    //loop over columns
	for(mwIndex j=0; j<n; j++){
		i=0;
		k=B.col[j];
		
		while((k<B.col[j+1])){
			while((i<B.row[k])&&(k<B.col[j+1])){
				if(get(i,j)!=0){
					C.get(i,j)=inf;
				}
				i++;
			}
			if(get(i,j)!=0){
				C.get(i,j)=get(i,j)/B.val[k];
			}
			i++;
			k++;
		}
		while(i<m){
			if(get(i,j)!=0){
				C.get(i,j)=inf;
			}
			i++;
		}
	}
	return C;
}

//by another full matrix
full full::operator / (const full &B) {
    
    //check appropriate size
	if(m!=B.m || n != B.n){
		mexErrMsgIdAndTxt("Pdiv:Dimension","Matrices must have same size");
	}

    //create output matrix
	full C(m,n);
    
	for(mwIndex i=0; i<m; i++){
		for(mwIndex j=0; j<n; j++){
			if(get(i,j)==0){
				C.get(i,j)=0;
			}
			else{
				C.get(i,j)=get(i,j)/B.get(i,j);
			}
		}
	}
	
	return C;
}
	


