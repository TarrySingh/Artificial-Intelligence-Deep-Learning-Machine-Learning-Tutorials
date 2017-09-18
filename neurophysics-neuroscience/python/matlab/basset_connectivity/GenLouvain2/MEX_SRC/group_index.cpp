//
//  group_index.cpp
//  group_index
//
//  Created by Lucas Jeub on 24/10/2012.
//
//  Implements the group_index datastructure:
//
//      nodes: vector storing the group memebership for each node
//
//      groups: vector of lists, each list stores the nodes assigned to the group
//
//      nodes_iterator: vector storing the position of each node in the list corresponding
//                      to the group it is assigned to (allows constant time moving of nodes)
//
//      index(group): return matlab indeces of nodes in group
//
//      move(node,group): move node to group
//
//      export_matlab(matlab_array): output group vector to matlab_array
//
//
//
//  Last modified by Lucas Jeub on 20/02/2013

#include "group_index.h"

using namespace std;

group_index::group_index():n_nodes(0), n_groups(0){}

group_index::group_index(const mxArray *matrix){
	mwSize m=mxGetM(matrix);
	mwSize n=mxGetN(matrix);
	n_nodes=m*n;
	double * temp_nodes = mxGetPr(matrix);
	
	nodes.resize(n_nodes);
	nodes_iterator.resize(n_nodes);
	for(mwIndex i=0; i<n_nodes; i++){
		nodes[i]=(mwIndex) temp_nodes[i]-1;
	}
	
	n_groups= * max_element(nodes.begin(),nodes.end())+1;
    
	groups.resize(n_groups);
	
	//initialise nodes
	for(mwIndex i=0; i<n_nodes; i++){
		groups[nodes[i]].push_back(i);
		nodes_iterator[i]= --groups[nodes[i]].end();
	}
}

group_index & group_index::operator=(const mxArray *group_vec){
    mwSize m=mxGetM(group_vec);
    mwSize n=mxGetN(group_vec);
    n_nodes=m*n;
    double * temp_nodes = mxGetPr(group_vec);
    
    nodes.clear();
    nodes.resize(n_nodes);
    nodes_iterator.clear();
    nodes_iterator.resize(n_nodes);
    
    for (mwIndex i=0; i<n_nodes; i++) {
        nodes[i]=(mwIndex) temp_nodes[i]-1;
    }
    
    n_groups = * max_element(nodes.begin(), nodes.end())+1;
    
    groups.clear();
    groups.resize(n_groups);
    
    
    for (mwIndex i=0; i<n_nodes; i++) {
        groups[nodes[i]].push_back(i);
        nodes_iterator[i]= --groups[nodes[i]].end();
    }
    
    return *this;
}


//return index of nodes in group
full group_index::index(mwIndex group){
    if (group<n_groups) {
    	full ind(1,groups[group].size());
        
        //iterate over elemnts in the list (add 1 for matlab indeces)
        mwIndex i=0;
        for(list<mwIndex>::iterator it=groups[group].begin(); it != groups[group].end(); it++){
            ind.get(i)=*it+1;
            i++;
        }
        return ind;
    }
    else {
        mexErrMsgIdAndTxt("group_index:index", "group number out of bounds");
    }
}

//moves node to specified group
void group_index::move(mwIndex node, mwIndex group){
    //move node by splicing into list for new group
	groups[group].splice(groups[group].end(), groups[nodes[node]],nodes_iterator[node]);
    //update its position
	nodes_iterator[node]= --groups[group].end();
    //update its group asignment
	nodes[node]=group;
}

void group_index::export_matlab(mxArray * & out){
    //implements tidyconfig
	out=mxCreateDoubleMatrix(n_nodes,1,mxREAL);
	double * val=mxGetPr(out);
    //keep track of nodes that have already been assigned
    vector<bool> track_move(n_nodes,true);
    mwIndex g_n=1;
	list<mwIndex>::iterator it;
	for(mwIndex i=0; i<n_nodes; i++){
		if(track_move[i]){
			for(it=groups[nodes[i]].begin(); it !=groups[nodes[i]].end();it++){
				val[*it]=g_n;
                track_move[*it]=false;
			}
            g_n++;
		}
	}	
}

