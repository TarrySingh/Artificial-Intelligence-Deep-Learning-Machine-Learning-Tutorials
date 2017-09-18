//
//  matlab_matrix.h
//  matlab_matrix
//
//  Created by Lucas Jeub on 24/10/2012
//
//  Implements thin wrapper classes for full and sparse matlab matrices
//
//
//  Last modified by Lucas Jeub on 25/07/2014






#ifndef MATLAB_MATRIX_H
#define MATLAB_MATRIX_H

#define inf std::numeric_limits<double>::infinity();


#include <limits>

#include "mex.h"

#ifndef OCTAVE
    #include "matrix.h"
#endif

struct full;

struct sparse{
	sparse();
	sparse(mwSize m, mwSize n, mwSize nmax);
	sparse(const sparse &matrix);
	sparse(const mxArray *matrix);

	~sparse();
	
	
	
	sparse & operator = (const sparse & matrix);
    
    sparse & operator = (const full & matrix);
	
	sparse & operator = (const mxArray *matrix);
	
	
	/*operations*/
	/*pointwise division*/

	sparse operator / (const sparse & B);
	sparse operator / (const full & B);

	mwSize nzero() const { return col[n];}
    
    double get(mwIndex i, mwIndex j);
	
	void export_matlab(mxArray * & out);
	
	mwSize m;
	mwSize n;
	mwSize nmax;
	mwIndex *row;
	mwIndex *col;
	double *val;

	private:
	
	bool export_flag;
};


struct full{
	full();
	full(mwSize m, mwSize n);
	full(const full &matrix);
	full(const mxArray * matrix);
	
	~full();
	
	void export_matlab(mxArray * & out);
	
	full & operator = (const full & matrix);
    
    full & operator = (const sparse & matrix);
	
	full & operator = (const mxArray * matrix);
	
	double & get(mwIndex i, mwIndex j);
    double get(mwIndex i,mwIndex j) const;
	double & get(mwIndex i);
    double get(mwIndex i) const;
    
	

	full operator / (const sparse &B);
	full operator / (const full &B);

	mwSize m;
	mwSize n;
	
	double *val;
	
	private:
	
	bool export_flag;
};
	

#endif
