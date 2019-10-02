/** \file
 * This is the standard tree adaptor used by the C runtime unless the grammar
 * source file says to use anything different. It embeds a BASE_TREE to which
 * it adds its own implementation of anything that the base tree is not 
 * good for, plus a number of methods that any other adaptor type
 * needs to implement too.
 * \ingroup pANTLR3_COMMON_TREE_ADAPTOR
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

#include    <antlr3commontreeadaptor.h>

#ifdef	ANTLR3_WINDOWS
#pragma warning( disable : 4100 )
#endif

/* BASE_TREE_ADAPTOR overrides... */
static	pANTLR3_BASE_TREE		dupNode					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE treeNode);
static	pANTLR3_BASE_TREE		create					(pANTLR3_BASE_TREE_ADAPTOR adpator, pANTLR3_COMMON_TOKEN payload);
static	pANTLR3_BASE_TREE		dbgCreate				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_COMMON_TOKEN payload);
static	pANTLR3_COMMON_TOKEN	createToken				(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text);
static	pANTLR3_COMMON_TOKEN	createTokenFromToken	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_COMMON_TOKEN fromToken);
static	pANTLR3_COMMON_TOKEN    getToken				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	pANTLR3_STRING			getText					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	ANTLR3_UINT32			getType					(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	pANTLR3_BASE_TREE		getChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 i);
static	ANTLR3_UINT32			getChildCount			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	void					replaceChildren			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, pANTLR3_BASE_TREE t);
static	void					setDebugEventListener	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_DEBUG_EVENT_LISTENER debugger);
static  void					setChildIndex			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_INT32 i);
static  ANTLR3_INT32			getChildIndex			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static	void					setParent				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE child, pANTLR3_BASE_TREE parent);
static  void					setChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 i, pANTLR3_BASE_TREE child);
static	void					deleteChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 i);
static	pANTLR3_BASE_TREE		errorNode				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_TOKEN_STREAM ctnstream, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken, pANTLR3_EXCEPTION e);
/* Methods specific to each tree adaptor
 */
static	void			setTokenBoundaries		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken);
static	void			dbgSetTokenBoundaries	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken);
static	ANTLR3_MARKER   getTokenStartIndex		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);
static  ANTLR3_MARKER   getTokenStopIndex		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t);

static	void		ctaFree			(pANTLR3_BASE_TREE_ADAPTOR adaptor);

/** Create a new tree adaptor. Note that despite the fact that this is
 *  creating a new COMMON_TREE adaptor, we return the address of the
 *  BASE_TREE interface, as should any other adaptor that wishes to be 
 *  used as the tree element of a tree parse/build. It needs to be given the
 *  address of a valid string factory as we do not know what the originating
 *  input stream encoding type was. This way we can rely on just using
 *  the original input stream's string factory or one of the correct type
 *  which the user supplies us.
 */
ANTLR3_API pANTLR3_BASE_TREE_ADAPTOR
ANTLR3_TREE_ADAPTORNew(pANTLR3_STRING_FACTORY strFactory)
{
	pANTLR3_COMMON_TREE_ADAPTOR	cta;

	// First job is to create the memory we need for the tree adaptor interface.
	//
	cta	= (pANTLR3_COMMON_TREE_ADAPTOR) ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_COMMON_TREE_ADAPTOR)));

	if	(cta == NULL)
	{
		return	NULL;
	}

	// Memory is initialized, so initialize the base tree adaptor
	//
	antlr3BaseTreeAdaptorInit(&(cta->baseAdaptor), NULL);

	// Install our interface overrides.
	//
	cta->baseAdaptor.dupNode				=  (void * (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
													dupNode;
	cta->baseAdaptor.create					=  (void * (*) (pANTLR3_BASE_TREE_ADAPTOR, pANTLR3_COMMON_TOKEN))
													create;
	cta->baseAdaptor.createToken			=  
													createToken;
	cta->baseAdaptor.createTokenFromToken   =  
													createTokenFromToken;
	cta->baseAdaptor.setTokenBoundaries	    =  (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, pANTLR3_COMMON_TOKEN, pANTLR3_COMMON_TOKEN))
													setTokenBoundaries;
	cta->baseAdaptor.getTokenStartIndex	    =  (ANTLR3_MARKER  (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                    getTokenStartIndex;
	cta->baseAdaptor.getTokenStopIndex	    =  (ANTLR3_MARKER  (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                    getTokenStopIndex;
	cta->baseAdaptor.getText				=  (pANTLR3_STRING (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                    getText;
	cta->baseAdaptor.getType				=  (ANTLR3_UINT32  (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                    getType;
	cta->baseAdaptor.getChild				=  (void * (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, ANTLR3_UINT32))
                                                    getChild;
	cta->baseAdaptor.setChild				=  (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, ANTLR3_UINT32, void *))
                                                    setChild;
	cta->baseAdaptor.setParent				=  (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, void *))
                                                    setParent;
	cta->baseAdaptor.setChildIndex			=  (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, ANTLR3_UINT32))
                                                    setChildIndex;
	cta->baseAdaptor.deleteChild			=  (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, ANTLR3_UINT32))
                                                    deleteChild;
	cta->baseAdaptor.getChildCount			=  (ANTLR3_UINT32  (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                    getChildCount;
	cta->baseAdaptor.getChildIndex			=  (ANTLR3_INT32  (*) (pANTLR3_BASE_TREE_ADAPTOR, void *))
                                                    getChildIndex;
	cta->baseAdaptor.free					=  (void  (*) (pANTLR3_BASE_TREE_ADAPTOR))
                                                    ctaFree;
	cta->baseAdaptor.setDebugEventListener	=  
													setDebugEventListener;
	cta->baseAdaptor.replaceChildren		=  (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, ANTLR3_INT32, ANTLR3_INT32, void *))
                                                    replaceChildren;
	cta->baseAdaptor.errorNode				=  (void * (*) (pANTLR3_BASE_TREE_ADAPTOR, pANTLR3_TOKEN_STREAM, pANTLR3_COMMON_TOKEN, pANTLR3_COMMON_TOKEN, pANTLR3_EXCEPTION))
                                                    errorNode;

	// Install the super class pointer
	//
	cta->baseAdaptor.super	    = cta;

	// Install a tree factory for creating new tree nodes
	//
	cta->arboretum  = antlr3ArboretumNew(strFactory);

	// Install a token factory for imaginary tokens, these imaginary
	// tokens do not require access to the input stream so we can
	// dummy the creation of it, but they will need a string factory.
	//
	cta->baseAdaptor.tokenFactory						= antlr3TokenFactoryNew(NULL);
	cta->baseAdaptor.tokenFactory->unTruc.strFactory	= strFactory;

	// Allow the base tree adaptor to share the tree factory's string factory.
	//
	cta->baseAdaptor.strFactory	= strFactory;

	// Return the address of the base adaptor interface.
	//
	return  &(cta->baseAdaptor);
}

/// Debugging version of the tree adaptor (not normally called as generated code
/// calls setDebugEventListener instead which changes a normal token stream to
/// a debugging stream and means that a user's instantiation code does not need
/// to be changed just to debug with AW.
///
ANTLR3_API pANTLR3_BASE_TREE_ADAPTOR
ANTLR3_TREE_ADAPTORDebugNew(pANTLR3_STRING_FACTORY strFactory, pANTLR3_DEBUG_EVENT_LISTENER	debugger)
{
	pANTLR3_BASE_TREE_ADAPTOR	ta;

	// Create a normal one first
	//
	ta	= ANTLR3_TREE_ADAPTORNew(strFactory);
	
	if	(ta != NULL)
	{
		// Reinitialize as a debug version
		//
		antlr3BaseTreeAdaptorInit(ta, debugger);
		ta->create				= (void * (*) (pANTLR3_BASE_TREE_ADAPTOR, pANTLR3_COMMON_TOKEN))
									dbgCreate;
		ta->setTokenBoundaries	= (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, pANTLR3_COMMON_TOKEN, pANTLR3_COMMON_TOKEN))
									dbgSetTokenBoundaries;
	}

	return	ta;
}

/// Causes an existing common tree adaptor to become a debug version
///
static	void
setDebugEventListener	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_DEBUG_EVENT_LISTENER debugger)
{
	// Reinitialize as a debug version
	//
	antlr3BaseTreeAdaptorInit(adaptor, debugger);

	adaptor->create				= (void * (*) (pANTLR3_BASE_TREE_ADAPTOR, pANTLR3_COMMON_TOKEN))
                                    dbgCreate;
	adaptor->setTokenBoundaries	= (void   (*) (pANTLR3_BASE_TREE_ADAPTOR, void *, pANTLR3_COMMON_TOKEN, pANTLR3_COMMON_TOKEN))
                                    dbgSetTokenBoundaries;

}

static void
ctaFree(pANTLR3_BASE_TREE_ADAPTOR adaptor)
{
    pANTLR3_COMMON_TREE_ADAPTOR cta;

    cta	= (pANTLR3_COMMON_TREE_ADAPTOR)(adaptor->super);

    /* Free the tree factory we created
     */
    cta->arboretum->close(((pANTLR3_COMMON_TREE_ADAPTOR)(adaptor->super))->arboretum);

    /* Free the token factory we created
     */
    adaptor->tokenFactory->close(adaptor->tokenFactory);

    /* Free the super pointer, as it is this that was allocated
     * and is the common tree structure.
     */
    ANTLR3_FREE(adaptor->super);
}

/* BASE_TREE_ADAPTOR overrides */

static	pANTLR3_BASE_TREE
errorNode				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_TOKEN_STREAM ctnstream, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken, pANTLR3_EXCEPTION e)
{
	// Use the supplied common tree node stream to get another tree from the factory
	// TODO: Look at creating the erronode as in Java, but this is complicated by the
	// need to track and free the memory allocated to it, so for now, we just
	// want something in the tree that isn't a NULL pointer.
	//
	return adaptor->createTypeText(adaptor, ANTLR3_TOKEN_INVALID, (pANTLR3_UINT8)"Tree Error Node");

}

/** Duplicate the supplied node.
 */
static	pANTLR3_BASE_TREE
dupNode		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE treeNode)
{
	return  treeNode == NULL ? NULL : treeNode->dupNode(treeNode);
}

static	pANTLR3_BASE_TREE
create		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_COMMON_TOKEN payload)
{
    pANTLR3_BASE_TREE	ct;
    
    /* Create a new common tree as this is what this adaptor deals with
     */
    ct = ((pANTLR3_COMMON_TREE_ADAPTOR)(adaptor->super))->arboretum->newFromToken(((pANTLR3_COMMON_TREE_ADAPTOR)(adaptor->super))->arboretum, payload);

    /* But all adaptors return the pointer to the base interface.
     */
    return  ct;
}
static	pANTLR3_BASE_TREE
dbgCreate		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_COMMON_TOKEN payload)
{
	pANTLR3_BASE_TREE	ct;

	ct = create(adaptor, payload);
	adaptor->debugger->createNode(adaptor->debugger, ct);

	return ct;
}

/** Tell me how to create a token for use with imaginary token nodes.
 *  For example, there is probably no input symbol associated with imaginary
 *  token DECL, but you need to create it as a payload or whatever for
 *  the DECL node as in ^(DECL type ID).
 *
 *  If you care what the token payload objects' type is, you should
 *  override this method and any other createToken variant.
 */
static	pANTLR3_COMMON_TOKEN
createToken		(pANTLR3_BASE_TREE_ADAPTOR adaptor, ANTLR3_UINT32 tokenType, pANTLR3_UINT8 text)
{
    pANTLR3_COMMON_TOKEN    newToken;

    newToken	= adaptor->tokenFactory->newToken(adaptor->tokenFactory);

    if	(newToken != NULL)
    {	
		newToken->textState		= ANTLR3_TEXT_CHARP;
		newToken->tokText.chars = (pANTLR3_UCHAR)text;
		newToken->setType(newToken, tokenType);
		newToken->input				= adaptor->tokenFactory->input;
    }
    return  newToken;
}

/** Tell me how to create a token for use with imaginary token nodes.
 *  For example, there is probably no input symbol associated with imaginary
 *  token DECL, but you need to create it as a payload or whatever for
 *  the DECL node as in ^(DECL type ID).
 *
 *  This is a variant of createToken where the new token is derived from
 *  an actual real input token.  Typically this is for converting '{'
 *  tokens to BLOCK etc...  You'll see
 *
 *    r : lc='{' ID+ '}' -> ^(BLOCK[$lc] ID+) ;
 *
 *  If you care what the token payload objects' type is, you should
 *  override this method and any other createToken variant.
 *
 * NB: this being C it is not so easy to extend the types of creaeteToken.
 *     We will have to see if anyone needs to do this and add any variants to
 *     this interface.
 */
static	pANTLR3_COMMON_TOKEN
createTokenFromToken	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_COMMON_TOKEN fromToken)
{
    pANTLR3_COMMON_TOKEN    newToken;

    newToken	= adaptor->tokenFactory->newToken(adaptor->tokenFactory);
    
    if	(newToken != NULL)
    {
		// Create the text using our own string factory to avoid complicating
		// commontoken.
		//
		pANTLR3_STRING	text;

		newToken->toString  = fromToken->toString;

		if	(fromToken->textState == ANTLR3_TEXT_CHARP)
		{
			newToken->textState		= ANTLR3_TEXT_CHARP;
			newToken->tokText.chars	= fromToken->tokText.chars;
		}
		else
		{
			text						= fromToken->getText(fromToken);
			newToken->textState			= ANTLR3_TEXT_STRING;
			newToken->tokText.text	    = adaptor->strFactory->newPtr(adaptor->strFactory, text->chars, text->len);
		}

		newToken->setLine				(newToken, fromToken->getLine(fromToken));
		newToken->setTokenIndex			(newToken, fromToken->getTokenIndex(fromToken));
		newToken->setCharPositionInLine	(newToken, fromToken->getCharPositionInLine(fromToken));
		newToken->setChannel			(newToken, fromToken->getChannel(fromToken));
		newToken->setType				(newToken, fromToken->getType(fromToken));
    }

    return  newToken;
}

/* Specific methods for a TreeAdaptor */

/** Track start/stop token for subtree root created for a rule.
 *  Only works with CommonTree nodes.  For rules that match nothing,
 *  seems like this will yield start=i and stop=i-1 in a nil node.
 *  Might be useful info so I'll not force to be i..i.
 */
static	void
setTokenBoundaries	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken)
{
	ANTLR3_MARKER   start;
	ANTLR3_MARKER   stop;

	pANTLR3_COMMON_TREE	    ct;

	if	(t == NULL)
	{
		return;
	}

	if	( startToken != NULL)
	{
		start = startToken->getTokenIndex(startToken);
	}
	else
	{
		start = 0;
	}

	if	( stopToken != NULL)
	{
		stop = stopToken->getTokenIndex(stopToken);
	}
	else
	{
		stop = 0;
	}

	ct	= (pANTLR3_COMMON_TREE)(t->super);

	ct->startIndex  = start;
	ct->stopIndex   = stop;

}
static	void
dbgSetTokenBoundaries	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, pANTLR3_COMMON_TOKEN startToken, pANTLR3_COMMON_TOKEN stopToken)
{
	setTokenBoundaries(adaptor, t, startToken, stopToken);

	if	(t != NULL && startToken != NULL && stopToken != NULL)
	{
		adaptor->debugger->setTokenBoundaries(adaptor->debugger, t, startToken->getTokenIndex(startToken), stopToken->getTokenIndex(stopToken));
	}
}

static	ANTLR3_MARKER   
getTokenStartIndex	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
    return  ((pANTLR3_COMMON_TREE)(t->super))->startIndex;
}

static	ANTLR3_MARKER   
getTokenStopIndex	(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
    return  ((pANTLR3_COMMON_TREE)(t->super))->stopIndex;
}

static	pANTLR3_STRING
getText		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
    return  t->getText(t);
}

static	ANTLR3_UINT32
getType		(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
    return  t->getType(t);
}

static	void					
replaceChildren
(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, pANTLR3_BASE_TREE t)
{
	if	(parent != NULL)
	{
		parent->replaceChildren(parent, startChildIndex, stopChildIndex, t);
	}
}

static	pANTLR3_BASE_TREE
getChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 i)
{
	return t->getChild(t, i);
}
static  void
setChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 i, pANTLR3_BASE_TREE child)
{
	t->setChild(t, i, child);
}

static	void
deleteChild				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_UINT32 i)
{
	t->deleteChild(t, i);
}

static	ANTLR3_UINT32
getChildCount			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
	return t->getChildCount(t);
}

static  void
setChildIndex			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t, ANTLR3_INT32 i)
{
	t->setChildIndex(t, i);
}

static  ANTLR3_INT32
getChildIndex			(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE t)
{
	return t->getChildIndex(t);
}
static	void
setParent				(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_TREE child, pANTLR3_BASE_TREE parent)
{
	child->setParent(child, parent);
}
