/** Interface for an ANTLR3 common tree which is what gets
 *  passed around by the AST producing parser.
 */

#ifndef	_ANTLR3_COMMON_TREE_H
#define	_ANTLR3_COMMON_TREE_H

// [The "BSD licence"]
// Copyright (c) 2005-2009 Jim Idle, Temporal Wave LLC
// http://www.temporal-wave.com
// http://www.linkedin.com/in/jimidle
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include    <antlr3defs.h>
#include    <antlr3basetree.h>
#include    <antlr3commontoken.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANTLR3_COMMON_TREE_struct
{

	/// Not used by ANTLR, but if a super structure is created above
    /// this structure, it can be used to point to the start of the super
    /// structure, where additional data and function pointers can be stored.
    ///
    void					* super;

    /// Start token index that encases this tree
    ///
    ANTLR3_MARKER			  startIndex;

    /// End token that encases this tree
    ///
    ANTLR3_MARKER			  stopIndex;

    /// A single token, this is the payload for the tree
    ///
    pANTLR3_COMMON_TOKEN      token;

	/// Points to the node that has this node as a child.
	/// If this is NULL, then this is the root node.
	///
	pANTLR3_COMMON_TREE		  parent;

	/// What index is this particular node in the child list it
	/// belongs to?
	///
	ANTLR3_INT32			  childIndex;
	
	/// Pointer to the tree factory that manufactured this
	/// token. This can be used by duplication methods and so on
	/// to manufacture another auto-tracked common tree structure
	///
	pANTLR3_ARBORETUM	factory;

    /// An encapsulated BASE TREE structure (NOT a pointer)
    /// that performs a lot of the dirty work of node management
    /// To this we add just a few functions that are specific to the 
    /// payload. You can further abstract common tree so long
    /// as you always have a baseTree pointer in the top structure
    /// and copy it from the next one down. 
    /// So, lets say we have a structure JIMS_TREE. 
    /// It needs an ANTLR3_BASE_TREE that will support all the
    /// general tree duplication stuff.
    /// It needs a ANTLR3_COMMON_TREE structure embedded or completely
    /// provides the equivalent interface.
    /// It provides it's own methods and data.
    /// To create a new one of these, the function provided to
    /// the tree adaptor (see comments there) should allocate the
    /// memory for a new JIMS_TREE structure, then call
    /// antlr3InitCommonTree(<addressofembeddedCOMMON_TREE>)
    /// antlr3BaseTreeNew(<addressofBASETREE>)
    /// The interfaces for BASE_TREE and COMMON_TREE will then
    /// be initialized. You then call and you can override them or just init
    /// JIMS_TREE (note that the base tree in common tree will be ignored)
    /// just the top level base tree is used). Codegen will take care of the rest.
    /// 
    ANTLR3_BASE_TREE	    baseTree;
     
}
    ANTLR3_COMMON_TREE;

/// \brief ANTLR3 Tree factory interface to create lots of trees efficiently
///  rather than creating and freeing lots of little bits of memory.
///
typedef	struct ANTLR3_ARBORETUM_struct
{
    /// Pointers to the array of tokens that this factory has produced so far
    ///
    pANTLR3_COMMON_TREE    *pools;

    /// Current pool tokens we are allocating from
    ///
    ANTLR3_INT32			thisPool;

    /// The next token to throw out from the pool, will cause a new pool allocation
    ///  if this exceeds the available tokenCount
    ///
    ANTLR3_UINT32			nextTree;

    /// Trick to initialize tokens and their API quickly, we set up this token when the
    /// factory is created, then just copy the memory it uses into the new token.
    ///
    ANTLR3_COMMON_TREE	    unTruc;

    /// Pointer to a vector factory that is used to create child list vectors
    /// for any child nodes that need them. This means that we auto track the
    /// vectors and auto free them when we close the factory. It also means
    /// that all rewriting trees can use the same tree factory and the same
    /// vector factory and we do not dup any nodes unless we must do so
    /// explicitly because of context such as an empty rewrite stream and
    /// ->IMAGINARY[ID] so on. This makes memory tracking much simpler and
    /// tempts no errors.
    ///
    pANTLR3_VECTOR_FACTORY   vFactory;

    /// A resuse stack for reclaiming Nil nodes that were used in rewrites
    /// and are now dead. The nilNode() method will eat one of these before
    /// creating a new node.
    ///
    pANTLR3_STACK           nilStack;

    /// Pointer to a function that returns a new tree
    ///
    pANTLR3_BASE_TREE	    (*newTree)		(struct ANTLR3_ARBORETUM_struct * factory);
    pANTLR3_BASE_TREE	    (*newFromTree)	(struct ANTLR3_ARBORETUM_struct * factory, pANTLR3_COMMON_TREE tree);
    pANTLR3_BASE_TREE	    (*newFromToken)	(struct ANTLR3_ARBORETUM_struct * factory, pANTLR3_COMMON_TOKEN token);

    /// Pointer to a function the destroys the factory
    ///
    void		    (*close)	    (struct ANTLR3_ARBORETUM_struct * factory);
}
    ANTLR3_ARBORETUM;

#ifdef __cplusplus
}
#endif

#endif


