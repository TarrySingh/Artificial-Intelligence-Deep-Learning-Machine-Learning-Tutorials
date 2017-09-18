//
//  sparse.cpp
//  sparse
//
//  Created by Lucas Jeub on 24/10/2012
//
//  Implements thin wrapper class for sparse matlab matrices
//
//
//  Last modified by Lucas Jeub on 27/11/2012


#include "matlab_matrix.h"


//default constructor
sparse::sparse():m(0), n(0),nmax(0),row(NULL),val(NULL), export_flag(0){
col = (mwIndex *) mxCalloc(1, sizeof(mwIndex));
col[0]=0;
}


//copy constructor
sparse::sparse(const sparse &matrix): m(matrix.m), n(matrix.n), nmax(matrix.nmax), export_flag(0) {
    
    //allocate memory
    row=(mwIndex *) mxCalloc(nmax,sizeof(mwIndex));
    col=(mwIndex *) mxCalloc(n+1, sizeof(mwIndex));
    val=(double *) mxCalloc(nmax,sizeof(double));
    
    //copy row indeces and values
    for(mwIndex i=0; i<matrix.nzero(); i++){
        row[i]=matrix.row[i];
        val[i]=matrix.val[i];
    }
    
    //copy column indeces
    for(mwIndex i=0; i<n+1; i++){
        col[i]=matrix.col[i];
    }
}


//construct by size
sparse::sparse(mwSize m_,mwSize n_,mwSize nmax_):m(m_), n(n_), nmax(nmax_), export_flag(0) {
    
    //allocate memory
    row=(mwIndex *) mxCalloc(nmax,sizeof(mwIndex));
    col=(mwIndex *) mxCalloc(n+1,sizeof(mwIndex));
    val=(double *) mxCalloc(nmax, sizeof(double));
}


//construct from mxArray (not save, useful for input arguments that are not modified, otherwise use operator = )
sparse::sparse(const mxArray * matrix): m(mxGetM(matrix)), n(mxGetN(matrix)), export_flag(0){
	
    if(mxIsDouble(matrix)){
        if(mxIsSparse(matrix)){
            //input is sparse
            
            nmax=mxGetNzmax(matrix);
            row=mxGetIr(matrix);
            val=mxGetPr(matrix);
            col=mxGetJc(matrix);
            
            export_flag=1;
        }
        else{
            
            //input is full
            
            //get input values
            double *pr = mxGetPr(matrix);
            
            //find number of non-zero elements
            nmax=0;
            for(mwIndex i=0; i<m*n; i++){
                if(pr[i]!=0){
                    nmax++;
                }
            }
            
            //allocate memory
            row=(mwIndex *) mxCalloc(nmax,sizeof(mwIndex));
            col=(mwIndex *) mxCalloc(n+1, sizeof(mwIndex));
            val=(double *) mxCalloc(nmax,sizeof(double));
            
            //assign values (c tracks number of non-zero elements used to assign appropriate values to col)
            mwIndex c=0;
            for(mwIndex j=0; j<n;j++){
                col[j]=c;
                for(mwIndex i=0; i<m; i++){
                    if(pr[i+j*m]!=0){
                        row[c]=i;
                        val[c]=pr[i+j*m];
                        c++;
                    }
                }
            }
            col[n]=c;
        }
	}
	else{
        //wrong input
        mexErrMsgIdAndTxt("sparse:constructor", "mxArray must be either full or sparse matrix");
    }
}


//destructor (free memory if it has not been exported  to matlab array)
sparse::~sparse(){
    
	if(!export_flag){
		mxFree(row);
		mxFree(col);
		mxFree(val);
	}
}


//copy assignment
sparse & sparse::operator = (const sparse & matrix){
	
    //check for self assignment
    if(this != &matrix){
        //copy size
		m=matrix.m;
		n=matrix.n;
		nmax=matrix.nmax;
        
        //allocate memory
		row=(mwIndex *) mxRealloc(row, nmax*sizeof(mwIndex));
		col=(mwIndex *) mxRealloc(col, (n+1)*sizeof(mwIndex));
		val=(double *) mxRealloc(val, nmax*sizeof(double));
		
        //copy row index and values
		for(mwIndex i=0; i<matrix.nzero(); i++){
			row[i]=matrix.row[i];
			val[i]=matrix.val[i];
		}
	
        //copy column index
		for(mwIndex i=0; i<=n; i++){
			col[i]=matrix.col[i];
		}
		
	}
	
	return *this;
}


//convert from full
sparse & sparse::operator=(const full &matrix){
    //copy size
    m=matrix.m;
    n=matrix.n;
    
    //find number of non-zero elements
    nmax=0;
    for (mwIndex i=0; i<m*n; i++) {
        if (matrix.val[i]!=0) {
            nmax++;
        }
    }
    
    //allocate memory
    row=(mwIndex *) mxRealloc(row, nmax*sizeof(mwIndex));
    col=(mwIndex *) mxRealloc(col, (n+1)*sizeof(mwIndex));
    val=(double *)  mxRealloc(val, nmax*sizeof(double));
    
    //assign values (c tracks number of non-zero elements used to assign appropriate values to col)
    mwIndex c=0;
    for (mwIndex j=0; j<n; j++) {
        col[j]=c;
        for (mwIndex i=0; i<m; i++){
            if (matrix.get(i,j)!=0) {
                row[c]=i;
                val[c]=matrix.get(i, j);
                c++;
            }
        }
    }
    col[n]=c;
    
    return *this;
}


//copy from mxArray (full or sparse)
sparse & sparse::operator = (const mxArray *matrix){
	
    //get size of input
    m=mxGetM(matrix);
	n=mxGetN(matrix);
	
	
	if(mxIsDouble(matrix)){
        if(mxIsSparse(matrix)){
            //input is sparse
            
            //allocate memory
            nmax=mxGetNzmax(matrix);
            row=(mwIndex *) mxRealloc(row,nmax*sizeof(mwIndex));
            col=(mwIndex *) mxRealloc(col,(n+1)*sizeof(mwIndex));
            val=(double *) mxRealloc(val,nmax*sizeof(double));
            
            //get values of input matrix
            mwIndex *row_in=mxGetIr(matrix);
            mwIndex * col_in=mxGetJc(matrix);
            double * val_in=mxGetPr(matrix);
            
            //copy row index and values
            for(mwIndex i=0; i<nmax; i++){
                row[i]=row_in[i];
                val[i]=val_in[i];
            }
            
            //copy column index
            for(mwIndex i=0; i<=n; i++){
                col[i]=col_in[i];
            }
        }
        else{
            
            //input is full
            
            //get input values
            double *pr = mxGetPr(matrix);
            
            //find number of non-zero elements
            nmax=0;
            for(mwIndex i=0; i<m*n; i++){
                if(pr[i]!=0){
                    nmax++;
                }
            }
            
            //allocate memory
            row=(mwIndex *) mxRealloc(row,nmax*sizeof(mwIndex));
            col=(mwIndex *) mxRealloc(col,(n+1)*sizeof(mwIndex));
            val=(double *) mxRealloc(val,nmax*sizeof(double));
            
            //assign values (c tracks number of non-zero elements used to assign appropriate values to col)
            mwIndex c=0;
            for(mwIndex j=0; j<n; j++){
                col[j]=c;
                for(mwIndex i=0; i<m;i++){
                    if(pr[i+j*m]!=0){
                        row[c]=i;
                        val[c]=pr[i+j*m];
                        c++;
                    }
                }
            }
            col[n]=c;
        }
    }
	else{
        mexErrMsgIdAndTxt("sparse:assignment", "mxArray must be either full or sparse matrix");
    }
	
	return *this;
}


//export to sparse matlab mxArray (sets export flag to avoid freeing memory if used to set output argument)
void sparse::export_matlab(mxArray * & out){
	
    //create empty sparse matrix
	out=mxCreateSparse(0,0,0,mxREAL);
	
    //free index and value
	mxFree(mxGetIr(out));
	mxFree(mxGetJc(out));
	mxFree(mxGetPr(out));
	
    //assign index and value to output
	mxSetIr(out, row);
	mxSetJc(out, col);
	mxSetPr(out, val);
	
    //set sizes
	mxSetM(out,m);
	mxSetN(out,n);
	mxSetNzmax(out,nmax);
	
    //set export flag
	export_flag=1;
	
}


/*operations*/

/*pointwise division (note 0/0=0;)*/

// by another sparse matrix
sparse sparse::operator / (const sparse &B){
	
    //check appropriate size
	if(m!=B.m || n != B.n){
		mexErrMsgIdAndTxt("Pdiv:Dimension","Matrices must have same size");
	}
    
    //create output matrix
	sparse C(m, n, nmax);
	
    mwIndex j,k;
    
    //loop over columns
	for(mwIndex i=0; i<n; i++){
		
		j=col[i]; //first index in column i
		k=B.col[i]; //first index in column i
		C.col[i]=j; //column index in output is the same
        
        //while in column i
		while((j<col[i+1])&&(k<B.col[i+1])){
            while((row[j]>B.row[k])&&(k<B.col[i+1])){
				k++;
			}
           
			if((row[j]==B.row[k])&&(k<B.col[i+1])) //B has a non-zero element in this position
            {
				C.val[j]=val[j]/B.val[k];
				C.row[j]=row[j];
			}
			else{
                //B is zero in this position
				C.val[j]=inf;
				C.row[j]=row[j];
			}
			//check next element of A
			j++;
		}
        //remaining elements of A
		while(j<col[i+1]){
			C.val[j]=inf;
			C.row[j]=row[j];
			j++;
		}
	}
	C.col[n]=j;
	
	return C;
}

//by a full matrix
sparse sparse::operator / (const full &B){
	
    //check appropriate size
	if(m!=B.m || n != B.n){
		mexErrMsgIdAndTxt("Pdiv:Dimension","Matrices must have same size");
	}
    
    //create output matrix
	sparse C(m, n, nmax);
	
	//loop over columns
	for(mwIndex i=0; i<n; i++){
		C.col[i]=col[i];
        //and non-zero elements in that column
		for(mwIndex j=col[i]; j<col[i+1]; j++){
			C.row[j]=row[j];
			C.val[j]=val[j]/B.get(row[j],i);
		}
	}
	C.col[n]=col[n];
	
	
	return C;
}

double sparse::get(mwIndex i, mwIndex j){
    if (j<n&&i<m) {
        mwIndex it=col[j];
        while((it<col[j+1])&&(row[it]<i)){
            it++;
        }
        if (row[it]==i) {
            return val[it];
        }
        else{
            return 0;
        }
    }
    else {
        mexErrMsgIdAndTxt("sparse:get", "Index out of bounds");
    }
}

