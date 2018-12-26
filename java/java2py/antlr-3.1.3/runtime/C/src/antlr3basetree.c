#include    <antlr3basetree.h>

#ifdef	ANTLR3_WINDOWS
#pragma warning( disable : 4100 )
#endif

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

static void				*	getChild			(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 i);
static ANTLR3_UINT32		getChildCount		(pANTLR3_BASE_TREE tree);
static ANTLR3_UINT32		getCharPositionInLine
(pANTLR3_BASE_TREE tree);
static ANTLR3_UINT32		getLine				(pANTLR3_BASE_TREE tree);
static pANTLR3_BASE_TREE    
getFirstChildWithType
(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 type);
static void					addChild			(pANTLR3_BASE_TREE tree, pANTLR3_BASE_TREE child);
static void					addChildren			(pANTLR3_BASE_TREE tree, pANTLR3_LIST kids);
static void					replaceChildren		(pANTLR3_BASE_TREE parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, pANTLR3_BASE_TREE t);

static	void				freshenPACIndexesAll(pANTLR3_BASE_TREE tree);
static	void				freshenPACIndexes	(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 offset);

static void					setChild			(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 i, void * child);
static void				*	deleteChild			(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 i);
static void				*	dupTree				(pANTLR3_BASE_TREE tree);
static pANTLR3_STRING		toStringTree		(pANTLR3_BASE_TREE tree);


ANTLR3_API pANTLR3_BASE_TREE
antlr3BaseTreeNew(pANTLR3_BASE_TREE  tree)
{
	/* api */
	tree->getChild				= getChild;
	tree->getChildCount			= getChildCount;
	tree->addChild				= (void (*)(pANTLR3_BASE_TREE, void *))(addChild);
	tree->addChildren			= addChildren;
	tree->setChild				= setChild;
	tree->deleteChild			= deleteChild;
	tree->dupTree				= dupTree;
	tree->toStringTree			= toStringTree;
	tree->getCharPositionInLine	= getCharPositionInLine;
	tree->getLine				= getLine;
	tree->replaceChildren		= replaceChildren;
	tree->freshenPACIndexesAll	= freshenPACIndexesAll;
	tree->freshenPACIndexes		= freshenPACIndexes;
	tree->getFirstChildWithType	= (void *(*)(pANTLR3_BASE_TREE, ANTLR3_UINT32))(getFirstChildWithType);
	tree->children				= NULL;
	tree->strFactory			= NULL;

	/* Rest must be filled in by caller.
	*/
	return  tree;
}

static ANTLR3_UINT32	
getCharPositionInLine	(pANTLR3_BASE_TREE tree)
{
	return  0;
}

static ANTLR3_UINT32	
getLine	(pANTLR3_BASE_TREE tree)
{
	return  0;
}
static pANTLR3_BASE_TREE
getFirstChildWithType	(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 type)
{
	ANTLR3_UINT32   i;
	ANTLR3_UINT32   cs;

	pANTLR3_BASE_TREE	t;
	if	(tree->children != NULL)
	{
		cs	= tree->children->size(tree->children);
		for	(i = 0; i < cs; i++)
		{
			t = (pANTLR3_BASE_TREE) (tree->children->get(tree->children, i));
			if  (tree->getType(t) == type)
			{
				return  (pANTLR3_BASE_TREE)t;
			}
		}
	}
	return  NULL;
}



static void    *
getChild		(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 i)
{
	if	(      tree->children == NULL
		|| i >= tree->children->size(tree->children))
	{
		return NULL;
	}
	return  tree->children->get(tree->children, i);
}


static ANTLR3_UINT32
getChildCount	(pANTLR3_BASE_TREE tree)
{
	if	(tree->children == NULL)
	{
		return 0;
	}
	else
	{
		return	tree->children->size(tree->children);
	}
}

void	    
addChild (pANTLR3_BASE_TREE tree, pANTLR3_BASE_TREE child)
{
	ANTLR3_UINT32   n;
	ANTLR3_UINT32   i;

	if	(child == NULL)
	{
		return;
	}

	if	(child->isNilNode(child) == ANTLR3_TRUE)
	{
		if  (child->children != NULL && child->children == tree->children)
		{
			// TODO: Change to exception rather than ANTLR3_FPRINTF?
			//
			ANTLR3_FPRINTF(stderr, "ANTLR3: An attempt was made to add a child list to itself!\n");
			return;
		}

        // Add all of the children's children to this list
        //
        if (child->children != NULL)
        {
            if (tree->children == NULL)
            {
                // We are build ing the tree structure here, so we need not
                // worry about duplication of pointers as the tree node
                // factory will only clean up each node once. So we just
                // copy in the child's children pointer as the child is
                // a nil node (has not root itself).
                //
                tree->children = child->children;
                child->children = NULL;
                freshenPACIndexesAll(tree);
                
            }
            else
            {
                // Need to copy the children
                //
                n = child->children->size(child->children);

                for (i = 0; i < n; i++)
                {
                    pANTLR3_BASE_TREE entry;
                    entry = child->children->get(child->children, i);

                    // ANTLR3 lists can be sparse, unlike Array Lists
                    //
                    if (entry != NULL)
                    {
                        tree->children->add(tree->children, entry, (void (ANTLR3_CDECL *) (void *))child->free);
                    }
                }
            }
		}
	}
	else
	{
		// Tree we are adding is not a Nil and might have children to copy
		//
		if  (tree->children == NULL)
		{
			// No children in the tree we are adding to, so create a new list on
			// the fly to hold them.
			//
			tree->createChildrenList(tree);
		}

		tree->children->add(tree->children, child, (void (ANTLR3_CDECL *)(void *))child->free);
		
	}
}

/// Add all elements of the supplied list as children of this node
///
static void
addChildren	(pANTLR3_BASE_TREE tree, pANTLR3_LIST kids)
{
	ANTLR3_UINT32    i;
	ANTLR3_UINT32    s;

	s = kids->size(kids);
	for	(i = 0; i<s; i++)
	{
		tree->addChild(tree, (pANTLR3_BASE_TREE)(kids->get(kids, i+1)));
	}
}


static    void
setChild	(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 i, void * child)
{
	if	(tree->children == NULL)
	{
		tree->createChildrenList(tree);
	}
	tree->children->set(tree->children, i, child, NULL, ANTLR3_FALSE);
}

static void    *
deleteChild	(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 i)
{
	if	( tree->children == NULL)
	{
		return	NULL;
	}

	return  tree->children->remove(tree->children, i);
}

static void    *
dupTree		(pANTLR3_BASE_TREE tree)
{
	pANTLR3_BASE_TREE	newTree;
	ANTLR3_UINT32	i;
	ANTLR3_UINT32	s;

	newTree = tree->dupNode	    (tree);

	if	(tree->children != NULL)
	{
		s	    = tree->children->size  (tree->children);

		for	(i = 0; i < s; i++)
		{
			pANTLR3_BASE_TREE    t;
			pANTLR3_BASE_TREE    newNode;

			t   = (pANTLR3_BASE_TREE) tree->children->get(tree->children, i);

			if  (t!= NULL)
			{
				newNode	    = t->dupTree(t);
				newTree->addChild(newTree, newNode);
			}
		}
	}

	return newTree;
}

static pANTLR3_STRING
toStringTree	(pANTLR3_BASE_TREE tree)
{
	pANTLR3_STRING  string;
	ANTLR3_UINT32   i;
	ANTLR3_UINT32   n;
	pANTLR3_BASE_TREE   t;

	if	(tree->children == NULL || tree->children->size(tree->children) == 0)
	{
		return	tree->toString(tree);
	}

	/* Need a new string with nothing at all in it.
	*/
	string	= tree->strFactory->newRaw(tree->strFactory);

	if	(tree->isNilNode(tree) == ANTLR3_FALSE)
	{
		string->append8	(string, "(");
		string->appendS	(string, tree->toString(tree));
		string->append8	(string, " ");
	}
	if	(tree->children != NULL)
	{
		n = tree->children->size(tree->children);

		for	(i = 0; i < n; i++)
		{   
			t   = (pANTLR3_BASE_TREE) tree->children->get(tree->children, i);

			if  (i > 0)
			{
				string->append8(string, " ");
			}
			string->appendS(string, t->toStringTree(t));
		}
	}
	if	(tree->isNilNode(tree) == ANTLR3_FALSE)
	{
		string->append8(string,")");
	}

	return  string;
}

/// Delete children from start to stop and replace with t even if t is
/// a list (nil-root tree). Num of children can increase or decrease.
/// For huge child lists, inserting children can force walking rest of
/// children to set their child index; could be slow.
///
static void					
replaceChildren		(pANTLR3_BASE_TREE parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, pANTLR3_BASE_TREE newTree)
{
	ANTLR3_INT32	replacingHowMany;		// How many nodes will go away
	ANTLR3_INT32	replacingWithHowMany;	// How many nodes will replace them
	ANTLR3_INT32	numNewChildren;			// Tracking variable
	ANTLR3_INT32	delta;					// Difference in new vs existing count

	ANTLR3_INT32	i;
	ANTLR3_INT32	j;

	pANTLR3_VECTOR	newChildren;			// Iterator for whatever we are going to add in
	ANTLR3_BOOLEAN	freeNewChildren;		// Whether we created the iterator locally or reused it

	if	(parent->children == NULL)
	{
		ANTLR3_FPRINTF(stderr, "replaceChildren call: Indexes are invalid; no children in list for %s", parent->getText(parent)->chars);
		return;
	}

	// Either use the existing list of children in the supplied nil node, or build a vector of the
	// tree we were given if it is not a nil node, then we treat both situations exactly the same
	//
	if	(newTree->isNilNode(newTree))
	{
		newChildren = newTree->children;
		freeNewChildren = ANTLR3_FALSE;		// We must NO free this memory
	}
	else
	{
		newChildren = antlr3VectorNew(1);
		if	(newChildren == NULL)
		{
			ANTLR3_FPRINTF(stderr, "replaceChildren: out of memory!!");
			exit(1);
		}
		newChildren->add(newChildren, (void *)newTree, NULL);

		freeNewChildren = ANTLR3_TRUE;		// We must free this memory
	}

	// Initialize
	//
	replacingHowMany		= stopChildIndex - startChildIndex + 1;
	replacingWithHowMany	= newChildren->size(newChildren);
	delta					= replacingHowMany - replacingWithHowMany;
	numNewChildren			= newChildren->size(newChildren);

	// If it is the same number of nodes, then do a direct replacement
	//
	if	(delta == 0)
	{
		pANTLR3_BASE_TREE	child;

		// Same number of nodes
		//
		j	= 0;
		for	(i = startChildIndex; i <= stopChildIndex; i++)
		{
			child = (pANTLR3_BASE_TREE) newChildren->get(newChildren, j);
			parent->children->set(parent->children, i, child, NULL, ANTLR3_FALSE);
			child->setParent(child, parent);
			child->setChildIndex(child, i);
		}
	}
	else if (delta > 0)
	{
		ANTLR3_UINT32	indexToDelete;

		// Less nodes than there were before
		// reuse what we have then delete the rest
		//
		for	(j = 0; j < numNewChildren; j++)
		{
			parent->children->set(parent->children, startChildIndex + j, newChildren->get(newChildren, j), NULL, ANTLR3_FALSE);
		}

		// We just delete the same index position until done
		//
		indexToDelete = startChildIndex + numNewChildren;

		for	(j = indexToDelete; j <= (ANTLR3_INT32)stopChildIndex; j++)
		{
			parent->children->remove(parent->children, indexToDelete);
		}

		parent->freshenPACIndexes(parent, startChildIndex);
	}
	else
	{
		ANTLR3_UINT32 numToInsert;

		// More nodes than there were before
		// Use what we can, then start adding
		//
		for	(j = 0; j < replacingHowMany; j++)
		{
			parent->children->set(parent->children, startChildIndex + j, newChildren->get(newChildren, j), NULL, ANTLR3_FALSE);
		}

		numToInsert = replacingWithHowMany - replacingHowMany;

		for	(j = replacingHowMany; j < replacingWithHowMany; j++)
		{
			parent->children->add(parent->children, newChildren->get(newChildren, j), NULL);
		}

		parent->freshenPACIndexes(parent, startChildIndex);
	}

	if	(freeNewChildren == ANTLR3_TRUE)
	{
		ANTLR3_FREE(newChildren->elements);
		newChildren->elements = NULL;
		newChildren->size = 0;
		ANTLR3_FREE(newChildren);		// Will not free the nodes
	}
}

/// Set the parent and child indexes for all children of the
/// supplied tree.
///
static	void
freshenPACIndexesAll(pANTLR3_BASE_TREE tree)
{
	tree->freshenPACIndexes(tree, 0);
}

/// Set the parent and child indexes for some of the children of the
/// supplied tree, starting with the child at the supplied index.
///
static	void
freshenPACIndexes	(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 offset)
{
	ANTLR3_UINT32	count;
	ANTLR3_UINT32	c;

	count	= tree->getChildCount(tree);		// How many children do we have 

	// Loop from the supplied index and set the indexes and parent
	//
	for	(c = offset; c < count; c++)
	{
		pANTLR3_BASE_TREE	child;

		child = tree->getChild(tree, c);

		child->setChildIndex(child, c);
		child->setParent(child, tree);
	}
}

