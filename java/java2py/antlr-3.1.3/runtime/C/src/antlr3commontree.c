// \file
//
// Implementation of ANTLR3 CommonTree, which you can use as a
// starting point for your own tree. Though it is often easier just to tag things on
// to the user pointer in the tree unless you are building a different type
// of structure.
//

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

#include    <antlr3commontree.h>


static pANTLR3_COMMON_TOKEN getToken				(pANTLR3_BASE_TREE tree);
static pANTLR3_BASE_TREE    dupNode					(pANTLR3_BASE_TREE tree);
static ANTLR3_BOOLEAN	    isNilNode					(pANTLR3_BASE_TREE tree);
static ANTLR3_UINT32	    getType					(pANTLR3_BASE_TREE tree);
static pANTLR3_STRING	    getText					(pANTLR3_BASE_TREE tree);
static ANTLR3_UINT32	    getLine					(pANTLR3_BASE_TREE tree);
static ANTLR3_UINT32	    getCharPositionInLine	(pANTLR3_BASE_TREE tree);
static pANTLR3_STRING	    toString				(pANTLR3_BASE_TREE tree);
static pANTLR3_BASE_TREE	getParent				(pANTLR3_BASE_TREE tree);
static void					setParent				(pANTLR3_BASE_TREE tree, pANTLR3_BASE_TREE parent);
static void    				setChildIndex			(pANTLR3_BASE_TREE tree, ANTLR3_INT32 i);
static ANTLR3_INT32			getChildIndex			(pANTLR3_BASE_TREE tree);
static void					createChildrenList		(pANTLR3_BASE_TREE tree);
static void                 reuse                   (pANTLR3_BASE_TREE tree);

// Factory functions for the Arboretum
//
static void					newPool				(pANTLR3_ARBORETUM factory);
static pANTLR3_BASE_TREE    newPoolTree			(pANTLR3_ARBORETUM factory);
static pANTLR3_BASE_TREE    newFromTree			(pANTLR3_ARBORETUM factory, pANTLR3_COMMON_TREE tree);
static pANTLR3_BASE_TREE    newFromToken		(pANTLR3_ARBORETUM factory, pANTLR3_COMMON_TOKEN token);
static void					factoryClose		(pANTLR3_ARBORETUM factory);

ANTLR3_API pANTLR3_ARBORETUM
antlr3ArboretumNew(pANTLR3_STRING_FACTORY strFactory)
{
    pANTLR3_ARBORETUM   factory;

    // Allocate memory
    //
    factory	= (pANTLR3_ARBORETUM) ANTLR3_MALLOC((size_t)sizeof(ANTLR3_ARBORETUM));
    if	(factory == NULL)
    {
		return	NULL;
    }

	// Install a vector factory to create, track and free() any child
	// node lists.
	//
	factory->vFactory					= antlr3VectorFactoryNew(0);
	if	(factory->vFactory == NULL)
	{
		free(factory);
		return	NULL;
	}

    // We also keep a reclaim stack, so that any Nil nodes that are
    // orphaned are not just left in the pool but are reused, other wise
    // we create 6 times as many nilNodes as ordinary nodes and use loads of
    // memory. Perhaps at some point, the analysis phase will generate better
    // code and we won't need to do this here.
    //
    factory->nilStack       =  antlr3StackNew(0);

    // Install factory API
    //
    factory->newTree	    =  newPoolTree;
    factory->newFromTree    =  newFromTree;
    factory->newFromToken   =  newFromToken;
    factory->close			=  factoryClose;

    // Allocate the initial pool
    //
    factory->thisPool	= -1;
    factory->pools		= NULL;
    newPool(factory);

    // Factory space is good, we now want to initialize our cheating token
    // which one it is initialized is the model for all tokens we manufacture
    //
    antlr3SetCTAPI(&factory->unTruc);

    // Set some initial variables for future copying, including a string factory
    // that we can use later for converting trees to strings.
    //
	factory->unTruc.factory				= factory;
    factory->unTruc.baseTree.strFactory	= strFactory;

    return  factory;

}

static void
newPool(pANTLR3_ARBORETUM factory)
{
    // Increment factory count
    //
    factory->thisPool++;

    // Ensure we have enough pointers allocated
    //
    factory->pools = (pANTLR3_COMMON_TREE *)
					ANTLR3_REALLOC(	(void *)factory->pools,										// Current pools pointer (starts at NULL)
					(ANTLR3_UINT32)((factory->thisPool + 1) * sizeof(pANTLR3_COMMON_TREE *))	// Memory for new pool pointers
					);

    // Allocate a new pool for the factory
    //
    factory->pools[factory->thisPool]	=
			    (pANTLR3_COMMON_TREE) 
				ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_COMMON_TREE) * ANTLR3_FACTORY_POOL_SIZE));


    // Reset the counters
    //
    factory->nextTree	= 0;
  
    // Done
    //
    return;
}

static	pANTLR3_BASE_TREE    
newPoolTree	    (pANTLR3_ARBORETUM factory)
{
	pANTLR3_COMMON_TREE    tree;

    // If we have anything on the re claim stack, reuse that sucker first
    //
    tree = factory->nilStack->peek(factory->nilStack);

    if  (tree != NULL)
    {
        // Cool we got something we could reuse, it will have been cleaned up by
        // whatever put it back on the stack (for instance if it had a child vector,
        // that will have been cleared to hold zero entries and that vector will get reused too.
        // It is the basetree pointer that is placed on the stack of course
        //
        factory->nilStack->pop(factory->nilStack);
        return (pANTLR3_BASE_TREE)tree;

    }
	// See if we need a new tree pool before allocating a new tree
	//
	if	(factory->nextTree >= ANTLR3_FACTORY_POOL_SIZE)
	{
		// We ran out of tokens in the current pool, so we need a new pool
		//
		newPool(factory);
	}

	// Assuming everything went well - we are trying for performance here so doing minimal
	// error checking - then we can work out what the pointer is to the next commontree.
	//
	tree   = factory->pools[factory->thisPool] + factory->nextTree;
	factory->nextTree++;

	// We have our token pointer now, so we can initialize it to the predefined model.
	//
    antlr3SetCTAPI(tree);

    // Set some initial variables for future copying, including a string factory
    // that we can use later for converting trees to strings.
    //
	tree->factory				= factory;
    tree->baseTree.strFactory	= factory->unTruc.baseTree.strFactory;

	// The super points to the common tree so we must override the one used by
	// by the pre-built tree as otherwise we will always poitn to the same initial
	// common tree and we might spend 3 hours trying to debug why - this would never
	// happen to me of course! :-(
	//
	tree->baseTree.super	= tree;

	// And we are done
	//
	return  &(tree->baseTree);
}


static pANTLR3_BASE_TREE	    
newFromTree(pANTLR3_ARBORETUM factory, pANTLR3_COMMON_TREE tree)
{
	pANTLR3_BASE_TREE	newTree;

	newTree = factory->newTree(factory);

	if	(newTree == NULL)
	{
		return	NULL;
	}

	// Pick up the payload we had in the supplied tree
	//
	((pANTLR3_COMMON_TREE)(newTree->super))->token   = tree->token;
	newTree->u		    = tree->baseTree.u;							// Copy any user pointer

	return  newTree;
}

static pANTLR3_BASE_TREE	    
newFromToken(pANTLR3_ARBORETUM factory, pANTLR3_COMMON_TOKEN token)
{
	pANTLR3_BASE_TREE	newTree;

	newTree = factory->newTree(factory);

	if	(newTree == NULL)
	{
		return	NULL;
	}

	// Pick up the payload we had in the supplied tree
	//
	((pANTLR3_COMMON_TREE)(newTree->super))->token = token;

	return newTree;
}

static	void
factoryClose	    (pANTLR3_ARBORETUM factory)
{
	ANTLR3_INT32	    poolCount;

	// First close the vector factory that supplied all the child pointer
	// vectors.
	//
	factory->vFactory->close(factory->vFactory);

    if  (factory->nilStack !=  NULL)
    {
        factory->nilStack->free(factory->nilStack);
    }

	// We now JUST free the pools because the C runtime CommonToken based tree
	// cannot contain anything that was not made by this factory.
	//
	for	(poolCount = 0; poolCount <= factory->thisPool; poolCount++)
	{
		// We can now free this pool allocation
		//
		ANTLR3_FREE(factory->pools[poolCount]);
		factory->pools[poolCount] = NULL;
	}

	// All the pools are deallocated we can free the pointers to the pools
	// now.
	//
	ANTLR3_FREE(factory->pools);

	// Finally, we can free the space for the factory itself
	//
	ANTLR3_FREE(factory);
}


ANTLR3_API void 
antlr3SetCTAPI(pANTLR3_COMMON_TREE tree)
{
    // Init base tree
    //
    antlr3BaseTreeNew(&(tree->baseTree));

    // We need a pointer to ourselves for 
    // the payload and few functions that we
    // provide.
    //
    tree->baseTree.super    =  tree;

    // Common tree overrides

    tree->baseTree.isNilNode					= isNilNode;
    tree->baseTree.toString					= toString;
    tree->baseTree.dupNode					= (void *(*)(pANTLR3_BASE_TREE))(dupNode);
    tree->baseTree.getLine					= getLine;
    tree->baseTree.getCharPositionInLine	= getCharPositionInLine;
    tree->baseTree.toString					= toString;
    tree->baseTree.getType					= getType;
    tree->baseTree.getText					= getText;
    tree->baseTree.getToken					= getToken;
	tree->baseTree.getParent				= getParent;
	tree->baseTree.setParent				= setParent;
	tree->baseTree.setChildIndex			= setChildIndex;
	tree->baseTree.getChildIndex			= getChildIndex;
	tree->baseTree.createChildrenList		= createChildrenList;
    tree->baseTree.reuse                    = reuse;
	tree->baseTree.free						= NULL;	// Factory trees have no free function
	
	tree->baseTree.children	= NULL;

    tree->token				= NULL;	// No token as yet
    tree->startIndex		= 0;
    tree->stopIndex			= 0;
	tree->parent			= NULL;	// No parent yet
	tree->childIndex		= -1;

    return;
}

// --------------------------------------
// Non factory node constructors.
//

ANTLR3_API pANTLR3_COMMON_TREE
antlr3CommonTreeNew()
{
	pANTLR3_COMMON_TREE	tree;
	tree    = ANTLR3_MALLOC(sizeof(ANTLR3_COMMON_TREE));

	if	(tree == NULL)
	{
		return NULL;
	}

	antlr3SetCTAPI(tree);

	return tree;
}

ANTLR3_API pANTLR3_COMMON_TREE	    
antlr3CommonTreeNewFromToken(pANTLR3_COMMON_TOKEN token)
{
	pANTLR3_COMMON_TREE	newTree;

	newTree = antlr3CommonTreeNew();

	if	(newTree == NULL)
	{
		return	NULL;
	}

	//Pick up the payload we had in the supplied tree
	//
	newTree->token = token;
	return newTree;
}

/// Create a new vector for holding child nodes using the inbuilt
/// vector factory.
///
static void
createChildrenList  (pANTLR3_BASE_TREE tree)
{
	tree->children = ((pANTLR3_COMMON_TREE)(tree->super))->factory->vFactory->newVector(((pANTLR3_COMMON_TREE)(tree->super))->factory->vFactory);
}


static pANTLR3_COMMON_TOKEN 
getToken			(pANTLR3_BASE_TREE tree)
{
    // The token is the payload of the common tree or other implementor
    // so it is stored within ourselves, which is the super pointer.Note 
	// that whatever the actual token is, it is passed around by its pointer
	// to the common token implementation, which it may of course surround
	// with its own super structure.
    //
    return  ((pANTLR3_COMMON_TREE)(tree->super))->token;
}

static pANTLR3_BASE_TREE    
dupNode			(pANTLR3_BASE_TREE tree)
{
    // The node we are duplicating is in fact the common tree (that's why we are here)
    // so we use the super pointer to duplicate.
    //
    pANTLR3_COMMON_TREE	    theOld;
    
	theOld	= (pANTLR3_COMMON_TREE)(tree->super);

	// The pointer we return is the base implementation of course
    //
	return  theOld->factory->newFromTree(theOld->factory, theOld);
}

static ANTLR3_BOOLEAN	    
isNilNode			(pANTLR3_BASE_TREE tree)
{
	// This is a Nil tree if it has no payload (Token in our case)
	//
	if	(((pANTLR3_COMMON_TREE)(tree->super))->token == NULL)
	{
		return ANTLR3_TRUE;
	}
	else
	{
		return ANTLR3_FALSE;
	}
}

static ANTLR3_UINT32	    
getType			(pANTLR3_BASE_TREE tree)
{
	pANTLR3_COMMON_TREE    theTree;

	theTree = (pANTLR3_COMMON_TREE)(tree->super);

	if	(theTree->token == NULL)
	{
		return	0;
	}
	else
	{
		return	theTree->token->getType(theTree->token);
	}
}

static pANTLR3_STRING	    
getText			(pANTLR3_BASE_TREE tree)
{
	return	tree->toString(tree);
}

static ANTLR3_UINT32	    getLine			(pANTLR3_BASE_TREE tree)
{
	pANTLR3_COMMON_TREE	    cTree;
	pANTLR3_COMMON_TOKEN    token;

	cTree   = (pANTLR3_COMMON_TREE)(tree->super);

	token   = cTree->token;

	if	(token == NULL || token->getLine(token) == 0)
	{
		if  (tree->getChildCount(tree) > 0)
		{
			pANTLR3_BASE_TREE	child;

			child   = (pANTLR3_BASE_TREE)tree->getChild(tree, 0);
			return child->getLine(child);
		}
		return 0;
	}
	return  token->getLine(token);
}

static ANTLR3_UINT32	    getCharPositionInLine	(pANTLR3_BASE_TREE tree)
{
	pANTLR3_COMMON_TOKEN    token;

	token   = ((pANTLR3_COMMON_TREE)(tree->super))->token;

	if	(token == NULL || token->getCharPositionInLine(token) == -1)
	{
		if  (tree->getChildCount(tree) > 0)
		{
			pANTLR3_BASE_TREE	child;

			child   = (pANTLR3_BASE_TREE)tree->getChild(tree, 0);

			return child->getCharPositionInLine(child);
		}
		return 0;
	}
	return  token->getCharPositionInLine(token);
}

static pANTLR3_STRING	    toString			(pANTLR3_BASE_TREE tree)
{
	if  (tree->isNilNode(tree) == ANTLR3_TRUE)
	{
		pANTLR3_STRING  nilNode;

		nilNode	= tree->strFactory->newPtr(tree->strFactory, (pANTLR3_UINT8)"nil", 3);

		return nilNode;
	}

	return	((pANTLR3_COMMON_TREE)(tree->super))->token->getText(((pANTLR3_COMMON_TREE)(tree->super))->token);
}

static pANTLR3_BASE_TREE	
getParent				(pANTLR3_BASE_TREE tree)
{
	return & (((pANTLR3_COMMON_TREE)(tree->super))->parent->baseTree);
}

static void					
setParent				(pANTLR3_BASE_TREE tree, pANTLR3_BASE_TREE parent)
{
	((pANTLR3_COMMON_TREE)(tree->super))->parent = parent == NULL ? NULL : ((pANTLR3_COMMON_TREE)(parent->super))->parent;
}

static void    				
setChildIndex			(pANTLR3_BASE_TREE tree, ANTLR3_INT32 i)
{
	((pANTLR3_COMMON_TREE)(tree->super))->childIndex = i;
}
static	ANTLR3_INT32			
getChildIndex			(pANTLR3_BASE_TREE tree )
{
	return ((pANTLR3_COMMON_TREE)(tree->super))->childIndex;
}

/** Clean up any child vector that the tree might have, so it can be reused,
 *  then add it into the reuse stack.
 */
static void
reuse                   (pANTLR3_BASE_TREE tree)
{
    pANTLR3_COMMON_TREE	    cTree;

	cTree   = (pANTLR3_COMMON_TREE)(tree->super);

    if  (cTree->factory != NULL)
    {
        if  (cTree->baseTree.children != NULL)
        {
            cTree->baseTree.children->clear(cTree->baseTree.children);
        }
        cTree->factory->nilStack->push(cTree->factory->nilStack, tree, NULL);
    }
}
