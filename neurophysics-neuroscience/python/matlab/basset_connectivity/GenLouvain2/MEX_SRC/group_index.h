//
//  group_index.h
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
//
//      index(group): return matlab indeces of nodes in group
//
//      move(node,group): move node to group
//
//      export_matlab(matlab_array): output group vector to matlab_a
//
//
//  Last modified by Lucas Jeub on 25/07/2014

#ifndef GROUP_INDEX_H
#define GROUP_INDEX_H

#include <list>
#include <vector>
#include <algorithm>

//interface with matlab
#include "mex.h"

#ifndef OCTAVE
    #include "matrix.h"
#endif

#include "matlab_matrix.h"



struct group_index{
	group_index();
	group_index(const mxArray *matrix); //assign group index from matlab
    
    group_index & operator = (const mxArray * group_vec); //assign group index from matlab
		
	full index(mwIndex group); //index of all nodes in group
	
	void move(mwIndex node, mwIndex group); //move node to group

	void export_matlab(mxArray * & out); //output group vector to matlab

	mwSize n_nodes;
	mwSize n_groups;

	std::vector<std::list<mwIndex> > groups; //the index of each node in a group is stored in a linked list
	std::vector<std::list<mwIndex>::iterator> nodes_iterator; //stores the position of the node in the list for the group it belongs to
	std::vector<mwIndex> nodes; //stores the group a node belongs to


};

#endif	

	
