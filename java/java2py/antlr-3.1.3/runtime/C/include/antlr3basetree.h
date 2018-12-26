/// \file
/// Definition of the ANTLR3 base tree.
///

#ifndef	_ANTLR3_BASE_TREE_H
#define	_ANTLR3_BASE_TREE_H

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

#ifdef __cplusplus
extern "C" {
#endif

/// A generic tree implementation with no payload.  You must subclass to
/// actually have any user data.  ANTLR v3 uses a list of children approach
/// instead of the child-sibling approach in v2.  A flat tree (a list) is
/// an empty node whose children represent the list.  An empty (as in it does not
/// have payload itself), but non-null node is called "nil".
///
typedef	struct ANTLR3_BASE_TREE_struct
{

    /// Implementers of this interface sometimes require a pointer to their selves.
    ///
    void    *	    super;

    /// Generic void pointer allows the grammar programmer to attach any structure they
    /// like to a tree node, in many cases saving the need to create their own tree
    /// and tree adaptors. ANTLR does not use this pointer, but will copy it for you and so on.
    ///
    void    *	    u;

    /// The list of all the children that belong to this node. They are not part of the node
    /// as they belong to the common tree node that implements this.
    ///
    pANTLR3_VECTOR  children;

    /// This is used to store the current child index position while descending
    /// and ascending trees as the tree walk progresses.
    ///
    ANTLR3_MARKER   savedIndex;

    /// A string factory to produce strings for toString etc
    ///
    pANTLR3_STRING_FACTORY strFactory;

    /// A pointer to a function that returns the common token pointer
    /// for the payload in the supplied tree.
    ///
    pANTLR3_COMMON_TOKEN                (*getToken)			(struct ANTLR3_BASE_TREE_struct * tree);

    void				(*addChild)			(struct ANTLR3_BASE_TREE_struct * tree, void * child);

    void				(*addChildren)			(struct ANTLR3_BASE_TREE_struct * tree, pANTLR3_LIST kids);

    void    				(*createChildrenList)		(struct ANTLR3_BASE_TREE_struct * tree);

    void    *				(*deleteChild)			(struct ANTLR3_BASE_TREE_struct * tree, ANTLR3_UINT32 i);

    void				(*replaceChildren)		(struct ANTLR3_BASE_TREE_struct * parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, struct ANTLR3_BASE_TREE_struct * t);

    void    *				(*dupNode)			(struct ANTLR3_BASE_TREE_struct * dupNode);

    void    *				(*dupTree)			(struct ANTLR3_BASE_TREE_struct * tree);

    ANTLR3_UINT32			(*getCharPositionInLine)	(struct ANTLR3_BASE_TREE_struct * tree);

    void    *				(*getChild)			(struct ANTLR3_BASE_TREE_struct * tree, ANTLR3_UINT32 i);

    void    				(*setChildIndex)		(struct ANTLR3_BASE_TREE_struct * tree, ANTLR3_INT32 );

    ANTLR3_INT32			(*getChildIndex)		(struct ANTLR3_BASE_TREE_struct * tree );

    ANTLR3_UINT32			(*getChildCount)		(struct ANTLR3_BASE_TREE_struct * tree);

    struct ANTLR3_BASE_TREE_struct *    (*getParent)			(struct ANTLR3_BASE_TREE_struct * tree);

    void    				(*setParent)			(struct ANTLR3_BASE_TREE_struct * tree, struct ANTLR3_BASE_TREE_struct * parent);

    ANTLR3_UINT32			(*getType)			(struct ANTLR3_BASE_TREE_struct * tree);

    void    *				(*getFirstChildWithType)	(struct ANTLR3_BASE_TREE_struct * tree, ANTLR3_UINT32 type);

    ANTLR3_UINT32			(*getLine)			(struct ANTLR3_BASE_TREE_struct * tree);

    pANTLR3_STRING			(*getText)			(struct ANTLR3_BASE_TREE_struct * tree);

    ANTLR3_BOOLEAN			(*isNilNode)			(struct ANTLR3_BASE_TREE_struct * tree);

    void				(*setChild)			(struct ANTLR3_BASE_TREE_struct * tree, ANTLR3_UINT32 i, void * child);

    pANTLR3_STRING			(*toStringTree)			(struct ANTLR3_BASE_TREE_struct * tree);

    pANTLR3_STRING			(*toString)			(struct ANTLR3_BASE_TREE_struct * tree);

    void				(*freshenPACIndexesAll)		(struct ANTLR3_BASE_TREE_struct * tree);

    void				(*freshenPACIndexes)		(struct ANTLR3_BASE_TREE_struct * tree, ANTLR3_UINT32 offset);

    void                                (*reuse)                        (struct ANTLR3_BASE_TREE_struct * tree);

    void    				(*free)				(struct ANTLR3_BASE_TREE_struct * tree);

}
    ANTLR3_BASE_TREE;

#ifdef __cplusplus
}
#endif


#endif
