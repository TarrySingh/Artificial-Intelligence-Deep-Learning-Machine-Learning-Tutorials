/** \file
 * Definition of the ANTLR3 base tree adaptor.
 */

#ifndef	_ANTLR3_BASE_TREE_ADAPTOR_H
#define	_ANTLR3_BASE_TREE_ADAPTOR_H

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
#include    <antlr3collections.h>
#include    <antlr3string.h>
#include    <antlr3basetree.h>
#include    <antlr3commontoken.h>
#include	<antlr3debugeventlistener.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef	struct ANTLR3_BASE_TREE_ADAPTOR_struct
{
    /** Pointer to any enclosing structure/interface that
     *  contains this structure.
     */
    void							* super;

    /** We need a string factory for creating imaginary tokens, we take this
     *  from the stream we are supplied to walk.
     */
    pANTLR3_STRING_FACTORY			strFactory;

    /* And we also need a token factory for creating imaginary tokens
     * this is also taken from the input source.
     */
    pANTLR3_TOKEN_FACTORY			tokenFactory;

	/// If set to something other than NULL, then this structure is
	/// points to an instance of the debugger interface. In general, the
	/// debugger is only referenced internally in recovery/error operations
	/// so that it does not cause overhead by having to check this pointer
	/// in every function/method
	///
	pANTLR3_DEBUG_EVENT_LISTENER	debugger;

    void *	   		 		(*nilNode)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor);


    void *	   		 		(*dupTree)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * tree);
    void *	   		 		(*dupTreeTT)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, void * tree);

    void					(*addChild)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, void * child);
    void					(*addChildToken)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, pANTLR3_COMMON_TOKEN child);
    void					(*setParent)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * child, void * parent);

	void *					(*errorNode)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, pANTLR3_TOKEN_STREAM tnstream, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken, pANTLR3_EXCEPTION e);
	ANTLR3_BOOLEAN			(*isNilNode)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t);

    void *	    			(*becomeRoot)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * newRoot, void * oldRoot);

    void *	   			 	(*rulePostProcessing)	(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * root);

    void *	   			 	(*becomeRootToken)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * newRoot, void * oldRoot);

    void *	   		 		(*create)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, pANTLR3_COMMON_TOKEN payload);
    void *	   		 		(*createTypeToken)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken);
    void *	   				(*createTypeTokenText)	(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken, pANTLR3_UINT8 text);
    void *	    			(*createTypeText)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text);

    void *	    			(*dupNode)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * treeNode);

    ANTLR3_UINT32			(*getType)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t);

    void					(*setType)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, ANTLR3_UINT32 type);
    
    pANTLR3_STRING			(*getText)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t);

    void					(*setText)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, pANTLR3_STRING t);
    void					(*setText8)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, pANTLR3_UINT8 t);

    void *	    			(*getChild)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, ANTLR3_UINT32 i);
    void					(*setChild)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, ANTLR3_UINT32 i, void * child);
    void					(*deleteChild)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, ANTLR3_UINT32 i);
    void				    (*setChildIndex)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, ANTLR3_UINT32 i);
    ANTLR3_INT32		    (*getChildIndex)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t);

    ANTLR3_UINT32			(*getChildCount)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void *);

    ANTLR3_UINT32			(*getUniqueID)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void *);

    pANTLR3_COMMON_TOKEN    (*createToken)			(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text);
    pANTLR3_COMMON_TOKEN    (*createTokenFromToken)	(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, pANTLR3_COMMON_TOKEN fromToken);
    pANTLR3_COMMON_TOKEN    (*getToken)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t);  

    void					(*setTokenBoundaries)	(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken);

    ANTLR3_MARKER			(*getTokenStartIndex)	(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t);

    ANTLR3_MARKER			(*getTokenStopIndex)	(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * t);

	void					(*setDebugEventListener)(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, pANTLR3_DEBUG_EVENT_LISTENER debugger);

	/// Produce a DOT (see graphviz freeware suite) from a base tree
	///
	pANTLR3_STRING			(*makeDot)				(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * theTree);

	/// Replace from start to stop child index of parent with t, which might
	/// be a list.  Number of children may be different
	/// after this call.  
	///
	/// If parent is null, don't do anything; must be at root of overall tree.
	/// Can't replace whatever points to the parent externally.  Do nothing.
	///
	void					(*replaceChildren)		(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor, void * parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, void * t);

    void					(*free)					(struct ANTLR3_BASE_TREE_ADAPTOR_struct * adaptor);

}
    ANTLR3_TREE_ADAPTOR, *pANTLR3_TREE_ADAPTOR;
#ifdef __cplusplus
}
#endif

#endif
