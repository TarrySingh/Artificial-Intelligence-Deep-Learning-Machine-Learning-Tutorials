/** \file
 * Contains the base functions that all tree adaptors start with.
 * this implementation can then be overridden by any higher implementation.
 * 
 */

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

#include    <antlr3basetreeadaptor.h>

#ifdef	ANTLR3_WINDOWS
#pragma warning( disable : 4100 )
#endif

/* Interface functions
 */
static	pANTLR3_BASE_TREE	nilNode					(pANTLR3_BASE_TREE_ADAPTOR adaptor);
static	pANTLR3_BASE_TREE	dbgNil					(pANTLR3_BASE_TREE_ADAPTOR adaptor);
static	pANTLR3_BASE_TREE	dupTree					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	pANTLR3_BASE_TREE	dbgDupTree				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	pANTLR3_BASE_TREE	dupTreeTT				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_BASE_TREE parent);
static	void				addChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_BASE_TREE child);
static	void				dbgAddChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_BASE_TREE child);
static	pANTLR3_BASE_TREE	becomeRoot				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE newRoot, pANTLR3_BASE_TREE oldRoot);
static	pANTLR3_BASE_TREE	dbgBecomeRoot			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE newRoot, pANTLR3_BASE_TREE oldRoot);
static	pANTLR3_BASE_TREE	rulePostProcessing		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE root);
static	void				addChildToken			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN child);
static	void				dbgAddChildToken		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN child);
static	pANTLR3_BASE_TREE	becomeRootToken			(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * newRoot, pANTLR3_BASE_TREE oldRoot);
static	pANTLR3_BASE_TREE	dbgBecomeRootToken		(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * newRoot, pANTLR3_BASE_TREE oldRoot);
static	pANTLR3_BASE_TREE	createTypeToken			(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken);
static	pANTLR3_BASE_TREE	dbgCreateTypeToken		(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken);
static	pANTLR3_BASE_TREE	createTypeTokenText		(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken, pANTLR3_UINT8 text);
static	pANTLR3_BASE_TREE	dbgCreateTypeTokenText	(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken, pANTLR3_UINT8 text);
static	pANTLR3_BASE_TREE	createTypeText			(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text);
static	pANTLR3_BASE_TREE	dbgCreateTypeText		(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text);
static	ANTLR3_UINT32		getType					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	void				setType					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 type);
static	pANTLR3_STRING		getText					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	void				setText					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_STRING t);
static	void				setText8				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_UINT8 t);
static	pANTLR3_BASE_TREE	getChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 i);
static	ANTLR3_UINT32		getChildCount			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	ANTLR3_UINT32		getUniqueID				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	ANTLR3_BOOLEAN		isNilNode				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	pANTLR3_STRING		makeDot					(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * theTree);

/** Given a pointer to a base tree adaptor structure (which is usually embedded in the
 *  super class the implements the tree adaptor used in the parse), initialize its
 *  function pointers and so on.
 */
ANTLR3_API void
antlr3BaseTreeAdaptorInit(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_DEBUG_EVENT_LISTENER	debugger)
{
	// Initialize the interface
	//
	if	(debugger == NULL)
	{
		adaptor->nilNode				= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR)) 								
																				nilNode;
		adaptor->addChild				= (void   (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, void *))								
																				addChild;
		adaptor->becomeRoot				= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, void *))				
																				becomeRoot;
		adaptor->addChildToken			= (void   (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, pANTLR3_COMMON_TOKEN))	
																				addChildToken;
		adaptor->becomeRootToken		= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, void *))
																				becomeRootToken;
		adaptor->createTypeToken		= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, ANTLR3_UINT32, pANTLR3_COMMON_TOKEN))
																				createTypeToken;
		adaptor->createTypeTokenText	= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, ANTLR3_UINT32, pANTLR3_COMMON_TOKEN, pANTLR3_UINT8))
																				createTypeTokenText;
		adaptor->createTypeText			= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, ANTLR3_UINT32, pANTLR3_UINT8))
																				createTypeText;
		adaptor->dupTree				= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))		 				
																				dupTree;
	}
	else
	{
		adaptor->nilNode				= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR))
                                                                                dbgNil;
		adaptor->addChild				= (void   (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, void *))
                                                                                dbgAddChild;
		adaptor->becomeRoot				= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, void *))
																				dbgBecomeRoot;
		adaptor->addChildToken			= (void   (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, pANTLR3_COMMON_TOKEN))
                                                                                dbgAddChildToken;
		adaptor->becomeRootToken		= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, void *))
                                                                                dbgBecomeRootToken;
		adaptor->createTypeToken		= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, ANTLR3_UINT32, pANTLR3_COMMON_TOKEN))
                                                                                dbgCreateTypeToken;
		adaptor->createTypeTokenText	= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, ANTLR3_UINT32, pANTLR3_COMMON_TOKEN, pANTLR3_UINT8))
                                                                                dbgCreateTypeTokenText;
		adaptor->createTypeText			= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, ANTLR3_UINT32, pANTLR3_UINT8))
                                                                                dbgCreateTypeText;
		adaptor->dupTree				= (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                                                dbgDupTree;
		debugger->adaptor				= adaptor;
	}

	adaptor->dupTreeTT				=  (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, void *))
                                                                                dupTreeTT;
	adaptor->rulePostProcessing		=  (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                                                rulePostProcessing;
	adaptor->getType				=  (ANTLR3_UINT32 (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                                                getType;
	adaptor->setType				=  (void   (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, ANTLR3_UINT32))
																				setType;
	adaptor->getText				=  (pANTLR3_STRING (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                                                getText;
	adaptor->setText8				=  (void   (*)(pANTLR3_BASE_TREE_ADAPTOR, pANTLR3_UINT8))
																				setText8;
	adaptor->setText				=  (void   (*)(pANTLR3_BASE_TREE_ADAPTOR, pANTLR3_STRING))
                                                                                setText;
	adaptor->getChild				=  (void * (*)(pANTLR3_BASE_TREE_ADAPTOR, void *, ANTLR3_UINT32))
                                                                                getChild;
	adaptor->getChildCount			=  (ANTLR3_UINT32 (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                                                getChildCount;
	adaptor->getUniqueID			=  (ANTLR3_UINT32 (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                                                getUniqueID;
	adaptor->isNilNode				=  (ANTLR3_BOOLEAN (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                                                isNilNode;

	adaptor->makeDot				=  (pANTLR3_STRING  (*)(pANTLR3_BASE_TREE_ADAPTOR, void *))
																				makeDot;
	
	/* Remaining functions filled in by the caller.
	 */
	return;
}

static void
defineDotNodes(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * t, pANTLR3_STRING dotSpec )
{
	// How many nodes are we talking about?
	//
	int	nCount;
	int i;

	if	(t == NULL)
	{
		// No tree, so create a blank spec
		//
		dotSpec->append8(dotSpec, "n0[label=\"EMPTY TREE\"]\n");
		return;
	}

	// Count the nodes
	//
	nCount = adaptor->getChildCount(adaptor, t);

	if	(nCount == 0)
	{
		// This will already have been included as a child of another node
		// so there is nothing to add.
		//
		return;
	}

	// For each child of the current tree, define a node using the
	// memory address of the node to name it
	//
	for	(i = 0; i<nCount; i++)
	{
		pANTLR3_BASE_TREE child;
		char	buff[64];
		pANTLR3_STRING	text;
		int		j;

		// Pick up a pointer for the child
		//
		child = adaptor->getChild(adaptor, t, i);

		// Name the node
		//
		sprintf(buff, "\tn%p[label=\"", child);
		dotSpec->append8(dotSpec, buff);
		text = adaptor->getText(adaptor, child);
		for (j = 0; j < (ANTLR3_INT32)(text->len); j++)
		{
			if	(text->charAt(text, j) == '"')
			{
				dotSpec->append8(dotSpec, "\\\"");
			}
			else
			{
				dotSpec->addc(dotSpec, text->charAt(text, j));
			}
		}
		dotSpec->append8(dotSpec, "\"]\n");

		// And now define the children of this child (if any)
		//
		defineDotNodes(adaptor, child, dotSpec);
	}
	
	// Done
	//
	return;
}

static void
defineDotEdges(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * t, pANTLR3_STRING dotSpec)
{
	// How many nodes are we talking about?
	//
	int	nCount;
	int i;

	if	(t == NULL)
	{
		// No tree, so do nothing
		//
		return;
	}

	// Count the nodes
	//
	nCount = adaptor->getChildCount(adaptor, t);

	if	(nCount == 0)
	{
		// This will already have been included as a child of another node
		// so there is nothing to add.
		//
		return;
	}

	// For each child, define an edge from this parent, then process
	// and children of this child in the same way
	//
	for	(i=0; i<nCount; i++)
	{
		pANTLR3_BASE_TREE child;
		char	buff[128];

		// Next child
		//
		child	= adaptor->getChild(adaptor, t, i);

		// Create the edge relation
		//
		sprintf(buff, "\t\tn%p -> n%p\t\t// ",  t, child);
		dotSpec->append8(dotSpec, buff);

		// Document the relationship
		//
		dotSpec->appendS(dotSpec, adaptor->getText(adaptor, t));
		dotSpec->append8(dotSpec, " -> ");
		dotSpec->appendS(dotSpec, adaptor->getText(adaptor, child));
		dotSpec->append8(dotSpec, "\n");

		// Define edges for this child
		//
		defineDotEdges(adaptor, child, dotSpec);
	}

	// Done
	//
	return;
}

/// Produce a DOT specification for graphviz
//
static pANTLR3_STRING
makeDot	(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * theTree)
{
	// The string we are building up
	//
	pANTLR3_STRING		dotSpec;

	dotSpec = adaptor->strFactory->newStr
		
		(
			adaptor->strFactory,

			// Default look and feel
			//
			(pANTLR3_UINT8)
			"digraph {\n\n"
			"\tordering=out;\n"
			"\tranksep=.4;\n"
			"\tbgcolor=\"lightgrey\";  node [shape=box, fixedsize=false, fontsize=12, fontname=\"Helvetica-bold\", fontcolor=\"blue\"\n"
			"\twidth=.25, height=.25, color=\"black\", fillcolor=\"white\", style=\"filled, solid, bold\"];\n\n"
			"\tedge [arrowsize=.5, color=\"black\", style=\"bold\"]\n\n"
		);

	// First produce the node defintions
	//
	defineDotNodes(adaptor, theTree, dotSpec);
	dotSpec->append8(dotSpec, "\n");
	defineDotEdges(adaptor, theTree, dotSpec);
	
	// Terminate the spec
	//
	dotSpec->append8(dotSpec, "\n}");

	// Result
	//
	return dotSpec;
}


/** Create and return a nil tree node (no token payload)
 */
static	pANTLR3_BASE_TREE	
nilNode	    (pANTLR3_BASE_TREE_ADAPTOR adaptor)
{
	return	adaptor->create(adaptor, NULL);
}

static	pANTLR3_BASE_TREE	
dbgNil	    (pANTLR3_BASE_TREE_ADAPTOR adaptor)
{
	pANTLR3_BASE_TREE t;

	t = adaptor->create				(adaptor, NULL);
	adaptor->debugger->createNode	(adaptor->debugger, t);

	return	t;
}

/** Return a duplicate of the entire tree (implementation provided by the 
 *  BASE_TREE interface.)
 */
static	pANTLR3_BASE_TREE	
dupTree  (pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
	return	adaptor->dupTreeTT(adaptor, t, NULL);
}

pANTLR3_BASE_TREE
dupTreeTT			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_BASE_TREE parent)
{
	pANTLR3_BASE_TREE	newTree;
	pANTLR3_BASE_TREE	child;
	pANTLR3_BASE_TREE	newSubTree;
	ANTLR3_UINT32		n;
	ANTLR3_UINT32		i;

	if	(t == NULL)
	{
		return NULL;
	}
	newTree = t->dupNode(t);

	// Ensure new subtree root has parent/child index set
	//
	adaptor->setChildIndex		(adaptor, newTree, t->getChildIndex(t));
	adaptor->setParent			(adaptor, newTree, parent);
	n = adaptor->getChildCount	(adaptor, t);

	for	(i=0; i < n; i++)
	{
		child = adaptor->getChild		(adaptor, t, i);
		newSubTree = adaptor->dupTreeTT	(adaptor, child, t);
		adaptor->addChild				(adaptor, newTree, newSubTree);
	}
	return	newTree;
}

/// Sends the required debugging events for duplicating a tree
/// to the debugger.
///
static void
simulateTreeConstruction(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE tree)
{
	ANTLR3_UINT32		n;
	ANTLR3_UINT32		i;
	pANTLR3_BASE_TREE	child;

	// Send the create node event
	//
	adaptor->debugger->createNode(adaptor->debugger, tree);

	n = adaptor->getChildCount(adaptor, tree);
	for	(i = 0; i < n; i++)
	{
		child = adaptor->getChild(adaptor, tree, i);
		simulateTreeConstruction(adaptor, child);
		adaptor->debugger->addChild(adaptor->debugger, tree, child);
	}
}

pANTLR3_BASE_TREE
dbgDupTree		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE tree)
{
	pANTLR3_BASE_TREE t;

	// Call the normal dup tree mechanism first
	//
	t = adaptor->dupTreeTT(adaptor, tree, NULL);

	// In order to tell the debugger what we have just done, we now
	// simulate the tree building mechanism. THis will fire
	// lots of debugging events to the client and look like we
	// duped the tree..
	//
	simulateTreeConstruction(adaptor, t);

	return t;
}

/** Add a child to the tree t.  If child is a flat tree (a list), make all
 *  in list children of t. Warning: if t has no children, but child does
 *  and child isNilNode then it is ok to move children to t via
 *  t.children = child.children; i.e., without copying the array.  This
 *  is for construction and I'm not sure it's completely general for
 *  a tree's addChild method to work this way.  Make sure you differentiate
 *  between your tree's addChild and this parser tree construction addChild
 *  if it's not ok to move children to t with a simple assignment.
 */
static	void	
addChild (pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_BASE_TREE child)
{
	if	(t != NULL && child != NULL)
	{
		t->addChild(t, child);
	}
}
static	void	
dbgAddChild (pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_BASE_TREE child)
{
	if	(t != NULL && child != NULL)
	{
		t->addChild(t, child);
		adaptor->debugger->addChild(adaptor->debugger, t, child);
	}
}
/** Use the adaptor implementation to add a child node with the supplied token
 */
static	void		
addChildToken		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN child)
{
	if	(t != NULL && child != NULL)
	{
		adaptor->addChild(adaptor, t, adaptor->create(adaptor, child));
	}
}
static	void		
dbgAddChildToken		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN child)
{
	pANTLR3_BASE_TREE	tc;

	if	(t != NULL && child != NULL)
	{
		tc = adaptor->create(adaptor, child);
		adaptor->addChild(adaptor, t, tc);
		adaptor->debugger->addChild(adaptor->debugger, t, tc);
	}
}

/** If oldRoot is a nil root, just copy or move the children to newRoot.
 *  If not a nil root, make oldRoot a child of newRoot.
 *
 * \code
 *    old=^(nil a b c), new=r yields ^(r a b c)
 *    old=^(a b c), new=r yields ^(r ^(a b c))
 * \endcode
 *
 *  If newRoot is a nil-rooted single child tree, use the single
 *  child as the new root node.
 *
 * \code
 *    old=^(nil a b c), new=^(nil r) yields ^(r a b c)
 *    old=^(a b c), new=^(nil r) yields ^(r ^(a b c))
 * \endcode
 *
 *  If oldRoot was null, it's ok, just return newRoot (even if isNilNode).
 *
 * \code
 *    old=null, new=r yields r
 *    old=null, new=^(nil r) yields ^(nil r)
 * \endcode
 *
 *  Return newRoot.  Throw an exception if newRoot is not a
 *  simple node or nil root with a single child node--it must be a root
 *  node.  If newRoot is <code>^(nil x)</endcode> return x as newRoot.
 *
 *  Be advised that it's ok for newRoot to point at oldRoot's
 *  children; i.e., you don't have to copy the list.  We are
 *  constructing these nodes so we should have this control for
 *  efficiency.
 */
static	pANTLR3_BASE_TREE	
becomeRoot	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE newRootTree, pANTLR3_BASE_TREE oldRootTree)
{
    pANTLR3_BASE_TREE saveRoot;

	/* Protect against tree rewrites if we are in some sort of error
	 * state, but have tried to recover. In C we can end up with a null pointer
	 * for a tree that was not produced.
	 */
	if	(newRootTree == NULL)
	{
		return	oldRootTree;
	}

	/* root is just the new tree as is if there is no
	 * current root tree.
	 */
	if	(oldRootTree == NULL)
	{
		return	newRootTree;
	}

	/* Produce ^(nil real-node)
	 */
	if	(newRootTree->isNilNode(newRootTree))
	{
		if	(newRootTree->getChildCount(newRootTree) > 1)
		{
			/* TODO: Handle tree exceptions 
			 */
			ANTLR3_FPRINTF(stderr, "More than one node as root! TODO: Create tree exception handling\n");
			return newRootTree;
		}

		/* The new root is the first child, keep track of the original newRoot
         * because if it was a Nil Node, then we can reuse it now.
		 */
        saveRoot    = newRootTree;
		newRootTree = newRootTree->getChild(newRootTree, 0);

        // Reclaim the old nilNode()
        //
        saveRoot->reuse(saveRoot);
	}

	/* Add old root into new root. addChild takes care of the case where oldRoot
	 * is a flat list (nill rooted tree). All children of oldroot are added to
	 * new root.
	 */
	newRootTree->addChild(newRootTree, oldRootTree);

    // If the oldroot tree was a nil node, then we know at this point
    // it has become orphaned by the rewrite logic, so we tell it to do
    // whatever it needs to do to be reused.
    //
    if  (oldRootTree->isNilNode(oldRootTree))
    {
        // We have taken an old Root Tree and appended all its children to the new
        // root. In addition though it was a nil node, which means the generated code
        // will not reuse it again, so we will reclaim it here. First we want to zero out
        // any pointers it was carrying around. We are just the baseTree handler so we
        // don't know necessarilly know how to do this for the real node, we just ask the tree itself
        // to do it.
        //
        oldRootTree->reuse(oldRootTree);
    }
	/* Always returns new root structure
	 */
	return	newRootTree;

}
static	pANTLR3_BASE_TREE	
dbgBecomeRoot	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE newRootTree, pANTLR3_BASE_TREE oldRootTree)
{
	pANTLR3_BASE_TREE t;
	
	t = becomeRoot(adaptor, newRootTree, oldRootTree);

	adaptor->debugger->becomeRoot(adaptor->debugger, newRootTree, oldRootTree);

	return t;
}
/** Transform ^(nil x) to x 
 */
static	pANTLR3_BASE_TREE	
   rulePostProcessing	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE root)
{
    pANTLR3_BASE_TREE saveRoot;

    // Keep track of the root we are given. If it is a nilNode, then we
    // can reuse it rather than orphaning it!
    //
    saveRoot = root;

	if (root != NULL && root->isNilNode(root))
	{
		if	(root->getChildCount(root) == 0)
		{
			root = NULL;
		}
		else if	(root->getChildCount(root) == 1)
		{
			root = root->getChild(root, 0);
			root->setParent(root, NULL);
			root->setChildIndex(root, -1);

            // The root we were given was a nil node, wiht one child, which means it has
            // been abandoned and would be lost in the node factory. However
            // nodes can be flagged as resuable to prevent this terrible waste
            //
            saveRoot->reuse(saveRoot);
		}
	}

	return root;
}
 
/** Use the adaptor interface to set a new tree node with the supplied token
 *  to the root of the tree.
 */
static	pANTLR3_BASE_TREE	
   becomeRootToken	(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * newRoot, pANTLR3_BASE_TREE oldRoot)
{
	return	adaptor->becomeRoot(adaptor, adaptor->create(adaptor, newRoot), oldRoot);
}
static	pANTLR3_BASE_TREE	
dbgBecomeRootToken	(pANTLR3_BASE_TREE_ADAPTOR adaptor, void * newRoot, pANTLR3_BASE_TREE oldRoot)
{
	pANTLR3_BASE_TREE	t;

	t =	adaptor->becomeRoot(adaptor, adaptor->create(adaptor, newRoot), oldRoot);

	adaptor->debugger->becomeRoot(adaptor->debugger,t, oldRoot);

	return t;
}

/** Use the super class supplied create() method to create a new node
 *  from the supplied token.
 */
static	pANTLR3_BASE_TREE	
createTypeToken	(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken)
{
	/* Create the new token
	 */
	fromToken = adaptor->createTokenFromToken(adaptor, fromToken);

	/* Set the type of the new token to that supplied
	 */
	fromToken->setType(fromToken, tokenType);

	/* Return a new node based upon this token
	 */
	return	adaptor->create(adaptor, fromToken);
}
static	pANTLR3_BASE_TREE	
dbgCreateTypeToken	(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken)
{
	pANTLR3_BASE_TREE t;

	t = createTypeToken(adaptor, tokenType, fromToken);

	adaptor->debugger->createNode(adaptor->debugger, t);

	return t;
}

static	pANTLR3_BASE_TREE	
createTypeTokenText	(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken, pANTLR3_UINT8 text)
{
	/* Create the new token
	 */
	fromToken = adaptor->createTokenFromToken(adaptor, fromToken);

	/* Set the type of the new token to that supplied
	 */
	fromToken->setType(fromToken, tokenType);

	/* Set the text of the token accordingly
	 */
	fromToken->setText8(fromToken, text);

	/* Return a new node based upon this token
	 */
	return	adaptor->create(adaptor, fromToken);
}
static	pANTLR3_BASE_TREE	
dbgCreateTypeTokenText	(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_COMMON_TOKEN fromToken, pANTLR3_UINT8 text)
{
	pANTLR3_BASE_TREE t;

	t = createTypeTokenText(adaptor, tokenType, fromToken, text);

	adaptor->debugger->createNode(adaptor->debugger, t);

	return t;
}

static	pANTLR3_BASE_TREE	
   createTypeText	(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text)
{
	pANTLR3_COMMON_TOKEN	fromToken;

	/* Create the new token
	 */
	fromToken = adaptor->createToken(adaptor, tokenType, text);

	/* Return a new node based upon this token
	 */
	return	adaptor->create(adaptor, fromToken);
}
static	pANTLR3_BASE_TREE	
   dbgCreateTypeText	(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text)
{
	pANTLR3_BASE_TREE t;

	t = createTypeText(adaptor, tokenType, text);

	adaptor->debugger->createNode(adaptor->debugger, t);

	return t;

}
/** Dummy implementation - will be supplied by super class
 */
static	ANTLR3_UINT32	
   getType		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
	return	0;
}

/** Dummy implementation - will be supplied by super class
 */
static	void		
   setType		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 type)
{
	ANTLR3_FPRINTF(stderr, "Internal error - implementor of superclass containing ANTLR3_TREE_ADAPTOR did not implement setType()\n");
}

/** Dummy implementation - will be supplied by super class
 */
static	pANTLR3_STRING	
   getText		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
	ANTLR3_FPRINTF(stderr, "Internal error - implementor of superclass containing ANTLR3_TREE_ADAPTOR did not implement getText()\n");
	return	NULL;
}

/** Dummy implementation - will be supplied by super class
 */
static	void		
   setText		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_STRING t)
{
	ANTLR3_FPRINTF(stderr, "Internal error - implementor of superclass containing ANTLR3_TREE_ADAPTOR did not implement setText()\n");
}
/** Dummy implementation - will be supplied by super class
 */
static	void		
setText8		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_UINT8 t)
{
	ANTLR3_FPRINTF(stderr, "Internal error - implementor of superclass containing ANTLR3_TREE_ADAPTOR did not implement setText()\n");
}

static	pANTLR3_BASE_TREE	
   getChild		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE tree, ANTLR3_UINT32 i)
{
	ANTLR3_FPRINTF(stderr, "Internal error - implementor of superclass containing ANTLR3_TREE_ADAPTOR did not implement getChild()\n");
	return NULL;
}

static	ANTLR3_UINT32	
   getChildCount	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE tree)
{
	ANTLR3_FPRINTF(stderr, "Internal error - implementor of superclass containing ANTLR3_TREE_ADAPTOR did not implement getChildCount()\n");
	return 0;
}

/** Returns a uniqueID for the node. Because this is the C implementation
 *  we can just use its address suitably converted/cast to an integer.
 */
static	ANTLR3_UINT32	
   getUniqueID		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE node)
{
	return	ANTLR3_UINT32_CAST(node);
}

static	ANTLR3_BOOLEAN
isNilNode					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
	return t->isNilNode(t);
}
