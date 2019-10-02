/// \file
/// Definition of the ANTLR3 common tree node stream.
///

#ifndef	_ANTLR3_COMMON_TREE_NODE_STREAM__H
#define	_ANTLR3_COMMON_TREE_NODE_STREAM__H

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
#include    <antlr3commontreeadaptor.h>
#include    <antlr3commontree.h>
#include    <antlr3collections.h>
#include    <antlr3intstream.h>
#include    <antlr3string.h>

/// Token buffer initial size settings ( will auto increase)
///
#define	DEFAULT_INITIAL_BUFFER_SIZE		100
#define	INITIAL_CALL_STACK_SIZE			10

#ifdef __cplusplus
extern "C" {
#endif

typedef	struct ANTLR3_TREE_NODE_STREAM_struct
{
    /// Any interface that implements this interface (is a 
    ///  super structure containing this structure), may store the pointer
    ///  to itself here in the super pointer, which is not used by 
    ///  the tree node stream. This will point to an implementation
    ///  of ANTLR3_COMMON_TREE_NODE_STREAM in this case.
    ///
    pANTLR3_COMMON_TREE_NODE_STREAM	ctns;

    /// All input streams implement the ANTLR3_INT_STREAM interface...
    ///
    pANTLR3_INT_STREAM			istream;

	/// Get tree node at current input pointer + i ahead where i=1 is next node.
	/// i<0 indicates nodes in the past.  So LT(-1) is previous node, but
	/// implementations are not required to provide results for k < -1.
	/// LT(0) is undefined.  For i>=n, return null.
	/// Return NULL for LT(0) and any index that results in an absolute address
	/// that is negative (beyond the start of the list).
	///
	/// This is analogous to the LT() method of the TokenStream, but this
	/// returns a tree node instead of a token.  Makes code gen identical
	/// for both parser and tree grammars. :)
	///
    pANTLR3_BASE_TREE			(*_LT)							(struct ANTLR3_TREE_NODE_STREAM_struct * tns, ANTLR3_INT32 k);

	/// Where is this stream pulling nodes from?  This is not the name, but
	/// the object that provides node objects.
	///
    pANTLR3_BASE_TREE			(*getTreeSource)				(struct ANTLR3_TREE_NODE_STREAM_struct * tns);

	/// What adaptor can tell me how to interpret/navigate nodes and
	/// trees.  E.g., get text of a node.
	///
    pANTLR3_BASE_TREE_ADAPTOR	(*getTreeAdaptor)				(struct ANTLR3_TREE_NODE_STREAM_struct * tns);

	/// As we flatten the tree, we use UP, DOWN nodes to represent
	/// the tree structure.  When debugging we need unique nodes
	/// so we have to instantiate new ones.  When doing normal tree
	/// parsing, it's slow and a waste of memory to create unique
	/// navigation nodes.  Default should be false;
	///
    void						(*setUniqueNavigationNodes)		(struct ANTLR3_TREE_NODE_STREAM_struct * tns, ANTLR3_BOOLEAN uniqueNavigationNodes);

    pANTLR3_STRING				(*toString)						(struct ANTLR3_TREE_NODE_STREAM_struct * tns);

	/// Return the text of all nodes from start to stop, inclusive.
	/// If the stream does not buffer all the nodes then it can still
	/// walk recursively from start until stop.  You can always return
	/// null or "" too, but users should not access $ruleLabel.text in
	/// an action of course in that case.
	///
    pANTLR3_STRING				(*toStringSS)					(struct ANTLR3_TREE_NODE_STREAM_struct * tns, pANTLR3_BASE_TREE start, pANTLR3_BASE_TREE stop);

	/// Return the text of all nodes from start to stop, inclusive, into the
	/// supplied buffer.
	/// If the stream does not buffer all the nodes then it can still
	/// walk recursively from start until stop.  You can always return
	/// null or "" too, but users should not access $ruleLabel.text in
	/// an action of course in that case.
	///
    void						(*toStringWork)					(struct ANTLR3_TREE_NODE_STREAM_struct * tns, pANTLR3_BASE_TREE start, pANTLR3_BASE_TREE stop, pANTLR3_STRING buf);

	/// Release up any and all space the the interface allocate, including for this structure.
	///
    void						(*free)							(struct ANTLR3_TREE_NODE_STREAM_struct * tns);

	/// Get a tree node at an absolute index i; 0..n-1.
	/// If you don't want to buffer up nodes, then this method makes no
	/// sense for you.
	///
	pANTLR3_BASE_TREE			(*get)							(struct ANTLR3_TREE_NODE_STREAM_struct * tns, ANTLR3_INT32 i);

	// REWRITING TREES (used by tree parser)

	/// Replace from start to stop child index of parent with t, which might
	/// be a list.  Number of children may be different
	/// after this call.  The stream is notified because it is walking the
	/// tree and might need to know you are monkeying with the underlying
	/// tree.  Also, it might be able to modify the node stream to avoid
	/// restreaming for future phases.
	///
	/// If parent is null, don't do anything; must be at root of overall tree.
	/// Can't replace whatever points to the parent externally.  Do nothing.
	///
	void						(*replaceChildren)				(struct ANTLR3_TREE_NODE_STREAM_struct * tns, pANTLR3_BASE_TREE parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, pANTLR3_BASE_TREE t);

}
    ANTLR3_TREE_NODE_STREAM;

typedef	struct ANTLR3_COMMON_TREE_NODE_STREAM_struct
{
    /// Any interface that implements this interface (is a 
    /// super structure containing this structure), may store the pointer
    /// to itself here in the super pointer, which is not used by 
    /// the common tree node stream.
    ///
    void						* super;

    /// Pointer to the tree node stream interface
    ///
    pANTLR3_TREE_NODE_STREAM	tnstream;

    /// String factory for use by anything that wishes to create strings
    /// such as a tree representation or some copy of the text etc.
    ///
    pANTLR3_STRING_FACTORY		stringFactory;

    /// Dummy tree node that indicates a descent into a child
    /// tree. Initialized by a call to create a new interface.
    ///
    ANTLR3_COMMON_TREE			DOWN;

    /// Dummy tree node that indicates a descent up to a parent
    /// tree. Initialized by a call to create a new interface.
    ///
    ANTLR3_COMMON_TREE			UP;

    /// Dummy tree node that indicates the termination point of the
    /// tree. Initialized by a call to create a new interface.
    ///
    ANTLR3_COMMON_TREE			EOF_NODE;

    /// Dummy node that is returned if we need to indicate an invalid node
    /// for any reason.
    ///
    ANTLR3_COMMON_TREE			INVALID_NODE;

	/// The complete mapping from stream index to tree node.
	/// This buffer includes pointers to DOWN, UP, and EOF nodes.
	/// It is built upon ctor invocation.  The elements are type
	/// Object as we don't what the trees look like.
	///
	/// Load upon first need of the buffer so we can set token types
	/// of interest for reverseIndexing.  Slows us down a wee bit to
	/// do all of the if p==-1 testing everywhere though, though in C
	/// you won't really be able to measure this.
	///
	/// Must be freed when the tree node stream is torn down.
	///
	pANTLR3_VECTOR				nodes;

    /// If set to ANTLR3_TRUE then the navigation nodes UP, DOWN are
    /// duplicated rather than reused within the tree.
    ///
    ANTLR3_BOOLEAN				uniqueNavigationNodes;

    /// Which tree are we navigating ?
    ///
    pANTLR3_BASE_TREE			root;

    /// Pointer to tree adaptor interface that manipulates/builds
    /// the tree.
    ///
    pANTLR3_BASE_TREE_ADAPTOR	adaptor;

    /// As we walk down the nodes, we must track parent nodes so we know
    /// where to go after walking the last child of a node.  When visiting
    /// a child, push current node and current index (current index
    /// is first stored in the tree node structure to avoid two stacks.
    ///
    pANTLR3_STACK				nodeStack;

	/// The current index into the nodes vector of the current tree
	/// we are parsing and possibly rewriting.
	///
	ANTLR3_INT32				p;

    /// Which node are we currently visiting?
    ///
    pANTLR3_BASE_TREE			currentNode;

    /// Which node did we last visit? Used for LT(-1)
    ///
    pANTLR3_BASE_TREE			previousNode;

    /// Which child are we currently visiting?  If -1 we have not visited
    /// this node yet; next consume() request will set currentIndex to 0.
    ///
    ANTLR3_INT32				currentChildIndex;

    /// What node index did we just consume?  i=0..n-1 for n node trees.
    /// IntStream.next is hence 1 + this value.  Size will be same.
    ///
    ANTLR3_MARKER				absoluteNodeIndex;

    /// Buffer tree node stream for use with LT(i).  This list grows
    /// to fit new lookahead depths, but consume() wraps like a circular
    /// buffer.
    ///
    pANTLR3_BASE_TREE	      * lookAhead;

    /// Number of elements available in the lookahead buffer at any point in
    ///  time. This is the current size of the array.
    ///
    ANTLR3_UINT32				lookAheadLength;

    /// lookAhead[head] is the first symbol of lookahead, LT(1). 
    ///
    ANTLR3_UINT32				head;

    /// Add new lookahead at lookahead[tail].  tail wraps around at the
    /// end of the lookahead buffer so tail could be less than head.
    ///
    ANTLR3_UINT32				tail;

    /// Calls to mark() may be nested so we have to track a stack of
    /// them.  The marker is an index into this stack.  Index 0 is
    /// the first marker.  This is a List<TreeWalkState>
    ///
    pANTLR3_VECTOR				markers;

    // INTERFACE
	//
    void				(*fill)						(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns, ANTLR3_INT32 k);

    void				(*addLookahead)				(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns, pANTLR3_BASE_TREE node);

    ANTLR3_BOOLEAN	    (*hasNext)					(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    pANTLR3_BASE_TREE	(*next)						(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    pANTLR3_BASE_TREE	(*handleRootnode)			(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    pANTLR3_BASE_TREE	(*visitChild)				(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns, ANTLR3_UINT32 child);

    void				(*addNavigationNode)		(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns, ANTLR3_UINT32 ttype);

    pANTLR3_BASE_TREE	(*newDownNode)				(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

	pANTLR3_BASE_TREE	(*newUpNode)				(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    void				(*walkBackToMostRecentNodeWithUnvisitedChildren)
													(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    ANTLR3_BOOLEAN	    (*hasUniqueNavigationNodes)	(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    ANTLR3_UINT32	    (*getLookaheadSize)			(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

	void				(*push)						(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns, ANTLR3_INT32 index);

	ANTLR3_INT32		(*pop)						(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    void				(*reset)					(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

    void				(*free)						(struct ANTLR3_COMMON_TREE_NODE_STREAM_struct * ctns);

	/// Indicates whether this node stream was derived from a prior
	/// node stream to be used by a rewriting tree parser for instance.
	/// If this flag is set to ANTLR3_TRUE, then when this stream is
	/// closed it will not free the root tree as this tree always
	/// belongs to the origniating node stream.
	///
	ANTLR3_BOOLEAN				isRewriter;

}
    ANTLR3_COMMON_TREE_NODE_STREAM;

/** This structure is used to save the state information in the treenodestream
 *  when walking ahead with cyclic DFA or for syntactic predicates,
 *  we need to record the state of the tree node stream.  This
 *  class wraps up the current state of the CommonTreeNodeStream.
 *  Calling mark() will push another of these on the markers stack.
 */
typedef struct ANTLR3_TREE_WALK_STATE_struct
{
    ANTLR3_UINT32			  currentChildIndex;
    ANTLR3_MARKER			  absoluteNodeIndex;
    pANTLR3_BASE_TREE		  currentNode;
    pANTLR3_BASE_TREE		  previousNode;
    ANTLR3_UINT32			  nodeStackSize;
    pANTLR3_BASE_TREE	    * lookAhead;
    ANTLR3_UINT32			  lookAheadLength;
    ANTLR3_UINT32			  tail;
    ANTLR3_UINT32			  head;
}
    ANTLR3_TREE_WALK_STATE;

#ifdef __cplusplus
}
#endif

#endif
