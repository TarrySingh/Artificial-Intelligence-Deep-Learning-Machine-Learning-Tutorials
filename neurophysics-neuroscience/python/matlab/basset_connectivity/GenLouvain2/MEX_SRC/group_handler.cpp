//
//  group_handler.cpp
//  group_handler
//
//  Created by Lucas Jeub on 21/11/2012.
//
// usage:
//
//  [output]=group_handler('function_handle',input)
//
//  implemented functions are 'assign', 'move', 'moverand', 'return'
//
//      assign: takes a group vector as input and uses it to initialise the "group_index"
//
//
//      move:   takes a node index and the corresponding column of the modularity matrix as
//              input
//
//              moves node to group with maximum improvement in modularity (stays in same group
//              if no improvement possible)
//
//              returns improvement if given an output argument
//
//
//      moverand:   takes a node index and the corresponding column of the modularity matrix as
//              input
//
//              moves node to random group with improvement in modularity (stays in same group
//              if no improvement possible)
//
//              returns improvement if given an output argument
//
//
//      return: outputs the community assignment for all nodes as a tidy group vector, that is
//              e.g. S = [1 2 1 3] rather than S = [2 4 2 6]
//
//
//  Last modified by Lucas Jeub on 23/03/2014



#include "group_handler.h"

using namespace std;

static group_index group;
default_random_engine generator((unsigned int)time(0));


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
                mexErrMsgIdAndTxt("group_handler:assign", "assign needs 1 input argument");
            }
            group=prhs[1];
        }
        else if (!strcmp(handle, "move")){
            if (nrhs!=3) {
                mexErrMsgIdAndTxt("group_handler:move", "move needs 2 input arguments");
            }
            mwIndex node=((mwIndex) * mxGetPr(prhs[1]))-1;
            double dstep=move(group, node, prhs[2]);
            
            //output improvement in modularity
            if (nlhs>0) {
                plhs[0]=mxCreateDoubleScalar(dstep);
            }
        }
		else if (!strcmp(handle, "moverand")){
            if (nrhs!=3) {
                mexErrMsgIdAndTxt("group_handler:moverand", "move needs 2 input arguments");
            }
            mwIndex node=((mwIndex) * mxGetPr(prhs[1]))-1;
            double dstep=moverand(group, node, prhs[2]);
            
            //output improvement in modularity
            if (nlhs>0) {
                plhs[0]=mxCreateDoubleScalar(dstep);
            }
        }
		else if (!strcmp(handle, "return")){
            if (nlhs>0) {
                group.export_matlab(plhs[0]);
            }
            else {
                mexErrMsgIdAndTxt("group_handler:return", "need ouput argument to return");
            }
        }
        else {
            mexErrMsgIdAndTxt("group_handler:handle", "invalid handle");
        }
    }
    else {
        mexErrMsgIdAndTxt("group_handler:input", "need handle to function");
    }
}


//move node to most optimal group
double move(group_index & g, mwIndex node, const mxArray * mod){
    set<mwIndex> unique_groups;
    unique_groups.insert(g.nodes[node]);
    
    map<mwIndex, double> mod_c; 
    
    if (mxIsSparse(mod)) {
        sparse mod_s(mod);
        
        //add nodes with potential positive contribution to unique_groups
        for(mwIndex j=0; j<mod_s.nzero(); j++){
            if(mod_s.val[j]>0){
                unique_groups.insert(g.nodes[mod_s.row[j]]);
            }
        }
        
        //calculate all changes in modularity
        mod_c=mod_change( g, mod_s, unique_groups, node);
    }
    else {
        full mod_d(mod);
        
        //add nodes with potential positive contribution to unique_groups
        for(mwIndex j=0; j<g.n_nodes; j++){
            if(mod_d.get(j)>0){
                unique_groups.insert(g.nodes[j]);
            }
        }
        
        //calculate all changes in modularity
        mod_c=mod_change(g, mod_d, unique_groups, node);
    }
    
    //find best move
    double mod_max=0;
    double d_step=0;

	mwIndex group_move=g.nodes[node]; //stay in current group if no improvement

	for(set<mwIndex>::iterator it=unique_groups.begin();it!=unique_groups.end();it++){
        if(mod_c[*it]>mod_max){
            mod_max=mod_c[*it];
            group_move=*it;
		}
    }

    //move current node to most optimal group
    if(mod_max>0){
        g.move(node,group_move);
        d_step+=mod_max;
    }

    return d_step;
}

//move node to random group increasing modularity
double moverand(group_index & g, mwIndex node, const mxArray * mod){
    set<mwIndex> unique_groups;
    unique_groups.insert(g.nodes[node]);
    
    map<mwIndex, double> mod_c; 
    
    if (mxIsSparse(mod)) {
        sparse mod_s(mod);
        
        //add nodes with potential positive contribution to unique_groups
        for(mwIndex j=0; j<mod_s.nzero(); j++){
            if(mod_s.val[j]>0){
                unique_groups.insert(g.nodes[mod_s.row[j]]);
            }
        }
        
        //calculate all changes in modularity
        mod_c=mod_change( g, mod_s, unique_groups, node);
    }
    else {
        full mod_d(mod);
        
        //add nodes with potential positive contribution to unique_groups
        for(mwIndex j=0; j<g.n_nodes; j++){
            if(mod_d.get(j)>0){
                unique_groups.insert(g.nodes[j]);
            }
        }
        
        //calculate all changes in modularity
        mod_c=mod_change(g, mod_d, unique_groups, node);
    }
    
    //find a random modularity increasing move
    double d_step=0;

	std::vector<mwIndex> unique_groups_pos; // groups that increase modularity
	std::vector<double>  mod_pos; // ammount by which they increase it
    
   

	// store groups that increase modularity
	bool notempty = false;
	for(set<mwIndex>::iterator it=unique_groups.begin();it!=unique_groups.end();it++){
		if(mod_c[*it]>0){
			notempty = true;
			unique_groups_pos.push_back(*it);
			mod_pos.push_back(mod_c[*it]);
		}
    }

	// move node to a random group that increases modularity
    uniform_int_distribution<mwIndex> randindex(0,unique_groups_pos.size()-1);
    
    mwIndex rand_move;
	if(notempty){
		//rand_move = rand() % unique_groups_pos.size();
        rand_move = randindex(generator);
        
        //mexPrintf("move %d, length %d \n",rand_move,unique_groups_pos.size());
	}
	
	if(notempty){ 
		g.move(node,unique_groups_pos[rand_move]);
        d_step+=mod_pos[rand_move];
	}

    return d_step;
}

//calculates changes in modularity for full modularity matrix
map<mwIndex, double> mod_change(group_index &g, full & mod, set<mwIndex> & unique_groups, mwIndex current_node){
    mwIndex current_group=g.nodes[current_node];
    map<mwIndex,double> mod_c;
    double mod_current= mod.get(current_node);
    
    for (set<mwIndex>::iterator it1=unique_groups.begin(); it1!=unique_groups.end(); it1++) {
        for(list<mwIndex>::iterator it2=g.groups[*it1].begin(); it2!=g.groups[*it1].end(); it2++){
            mod_c[*it1]+=mod.get(*it2);
        }
    }
    
    mod_c[current_group]-=mod_current;
    mod_current=mod_c[current_group];
    
    for (set<mwIndex>::iterator it=unique_groups.begin(); it!=unique_groups.end(); it++) {
        mod_c[*it]-=mod_current;
    }
    
    return mod_c;
}


//calculates changes in modularity for sparse modularity matrix
map<mwIndex,double> mod_change(group_index &g, sparse & mod, set<mwIndex> & unique_groups, mwIndex current_node){
    
    mwIndex current_group=g.nodes[current_node];
    map<mwIndex,double> mod_c;
    double mod_current=mod.get(current_node, 0);
    
    for(set<mwIndex>::iterator it=unique_groups.begin(); it!=unique_groups.end();it++){
        mod_c[*it]=0;
    }
    
    //calculate changes in modularity
    for (mwIndex i=0; i<mod.nzero(); i++) {
        if (unique_groups.count(g.nodes[mod.row[i]])) {
            mod_c[g.nodes[mod.row[i]]]+=mod.val[i];
        }
    }
    
    mod_c[current_group]-=mod_current;
    mod_current=mod_c[current_group];
    
    for (set<mwIndex>::iterator it=unique_groups.begin(); it!=unique_groups.end(); it++) {
        mod_c[*it]-=mod_current;
    }
    
    return mod_c;
}
