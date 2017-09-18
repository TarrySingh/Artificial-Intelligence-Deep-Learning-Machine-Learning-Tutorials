//
//  metanetwork_reduce.cpp
//  metanetwork_reduce
//
//  Created by Lucas Jeub on 22/11/2012.
//
// usage:
//
//  [output]=metanetwork_reduce('function_handle',input)
//
//  implemented functions are 'assign', 'move', 'return'
//
//      assign: takes a group vector as input and uses it to initialise the "group_index"
//
//
//      reduce: takes a column of the modularity matrix as
//              input
//
//              returns reduced column, where the i's entry is the sum of the original modularity
//              matrix over all nodes in group i
//
//
//      nodes: takes a group and returns the matlab index of all nodes in this group
//
//
//  Last modified by Lucas Jeub on 25/07/2014


#include "mex.h"


#include "matlab_matrix.h"
#include "group_index.h"
#include <map>
#include <cstring>

#ifndef OCTAVE
    #include "matrix.h"
#endif

using namespace std;
static group_index group;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if (nrhs>0) {
        //get handle to function to perform
        mwSize strleng = mxGetM(prhs[0])*mxGetN(prhs[0])+1;
        char * handle;
        handle=(char *) mxCalloc(strleng, sizeof(char));
        
        if (mxGetString(prhs[0],handle,strleng)) {
            mexErrMsgIdAndTxt("group_handler:handle", "handle needs to be a string");
        }
        
        //switch on handle
        if (!strcmp(handle, "assign")) {
            if (nrhs!=2) {
                mexErrMsgIdAndTxt("metanetwork_reduce:assign", "assign needs 1 input argument");
            }
            group=prhs[1];
        }
        else if (!strcmp(handle, "reduce")){
            if (nrhs!=2||nlhs<1) {
                mexErrMsgIdAndTxt("metanetwork_reduce:reduce", "reduce needs 1 input and 1 output argument");
            }
            if (mxIsDouble(prhs[1])) {
                full mod_out(group.n_groups,1);
                if (mxIsSparse(prhs[1])) {
                    sparse mod_s(prhs[1]);
                    for (mwIndex i=0; i<mod_s.nzero(); i++) {
                        mod_out.get(group.nodes[mod_s.row[i]])+=mod_s.val[i];
                    }
                }
                else {
                    full mod_d(prhs[1]);
                    for (mwIndex i=0; i<group.n_groups; i++) {
                        for (list<mwIndex>::iterator it=group.groups[i].begin(); it!=group.groups[i].end(); it++) {
                            mod_out.get(i)+=mod_d.get(*it);
                        }
                    }
                }
                mod_out.export_matlab(plhs[0]);
            }
        }
        else if (!strcmp(handle, "nodes")) {
            if (nrhs!=2||nlhs<1) {
                mexErrMsgIdAndTxt("metaneetwork_reduce:nodes", "nodes needs 1 input and 1 output argument");
            }
            full nodes=group.index(*mxGetPr(prhs[1])-1);
            nodes.export_matlab(plhs[0]);
        }
    }

}