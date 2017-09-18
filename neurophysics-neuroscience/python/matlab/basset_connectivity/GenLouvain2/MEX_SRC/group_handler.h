//
//  group_handler.h
//  group_handler
//
//  Created by Lucas Jeub on 21/11/2012.
//
//  Last modified by Lucas Jeub on 25/07/2014

#ifndef __group_handler__group_handler__
#define __group_handler__group_handler__

#include "mex.h"

#ifndef OCTAVE
    #include "matrix.h"
#endif

#include "matlab_matrix.h"
#include "group_index.h"
#include <cstring>
#include <map>
#include <set>
#include <vector>
#include <random>
#include <ctime>




//move node i to group with most improvement in modularity
double move(group_index & g, mwIndex node, const mxArray * mod);

//move node i to random group that increases modularity
double moverand(group_index & g, mwIndex node, const mxArray * mod);

std::map<mwIndex,double> mod_change(group_index &g, sparse &mod,std::set<mwIndex> & unique_groups,mwIndex current_node);

std::map<mwIndex, double> mod_change(group_index &g, full & mod, std::set<mwIndex> & unique_groups, mwIndex current_node);


#endif /* defined(__group_handler__group_handler__) */
