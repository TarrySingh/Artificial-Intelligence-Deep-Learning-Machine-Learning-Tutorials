/// \file
/// Defines the implementation of the common node stream the default
/// tree node stream used by ANTLR.
///

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

#include    <antlr3commontreenodestream.h>

#ifdef	ANTLR3_WINDOWS
#pragma warning( disable : 4100 )
#endif

// COMMON TREE STREAM API
//
static	void						addNavigationNode			(pANTLR3_COMMON_TREE_NODE_STREAM ctns, ANTLR3_UINT32 ttype);
static	ANTLR3_BOOLEAN				hasUniqueNavigationNodes	(pANTLR3_COMMON_TREE_NODE_STREAM ctns);
static	pANTLR3_BASE_TREE			newDownNode					(pANTLR3_COMMON_TREE_NODE_STREAM ctns);
static	pANTLR3_BASE_TREE			newUpNode					(pANTLR3_COMMON_TREE_NODE_STREAM ctns);
static	void						reset						(pANTLR3_COMMON_TREE_NODE_STREAM ctns);
static	void						push						(pANTLR3_COMMON_TREE_NODE_STREAM ctns, ANTLR3_INT32 index);
static	ANTLR3_INT32				pop							(pANTLR3_COMMON_TREE_NODE_STREAM ctns);
//static	ANTLR3_INT32				index						(pANTLR3_COMMON_TREE_NODE_STREAM ctns);
static	ANTLR3_UINT32				getLookaheadSize			(pANTLR3_COMMON_TREE_NODE_STREAM ctns);
// TREE NODE STREAM API
//
static	pANTLR3_BASE_TREE_ADAPTOR   getTreeAdaptor				(pANTLR3_TREE_NODE_STREAM tns);
static	pANTLR3_BASE_TREE			getTreeSource				(pANTLR3_TREE_NODE_STREAM tns);
static	pANTLR3_BASE_TREE			_LT							(pANTLR3_TREE_NODE_STREAM tns, ANTLR3_INT32 k);
static	pANTLR3_BASE_TREE			get							(pANTLR3_TREE_NODE_STREAM tns, ANTLR3_INT32 k);
static	void						setUniqueNavigationNodes	(pANTLR3_TREE_NODE_STREAM tns, ANTLR3_BOOLEAN uniqueNavigationNodes);
static	pANTLR3_STRING				toString					(pANTLR3_TREE_NODE_STREAM tns);
static	pANTLR3_STRING				toStringSS					(pANTLR3_TREE_NODE_STREAM tns, pANTLR3_BASE_TREE start, pANTLR3_BASE_TREE stop);
static	void						toStringWork				(pANTLR3_TREE_NODE_STREAM tns, pANTLR3_BASE_TREE start, pANTLR3_BASE_TREE stop, pANTLR3_STRING buf);
static	void						replaceChildren				(pANTLR3_TREE_NODE_STREAM tns, pANTLR3_BASE_TREE parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, pANTLR3_BASE_TREE t);

// INT STREAM API
//
static	void						consume						(pANTLR3_INT_STREAM is);
static	ANTLR3_MARKER				tindex						(pANTLR3_INT_STREAM is);
static	ANTLR3_UINT32				_LA							(pANTLR3_INT_STREAM is, ANTLR3_INT32 i);
static	ANTLR3_MARKER				mark						(pANTLR3_INT_STREAM is);
static	void						release						(pANTLR3_INT_STREAM is, ANTLR3_MARKER marker);
static	void						rewindMark					(pANTLR3_INT_STREAM is, ANTLR3_MARKER marker);
static	void						rewindLast					(pANTLR3_INT_STREAM is);
static	void						seek						(pANTLR3_INT_STREAM is, ANTLR3_MARKER index);
static	ANTLR3_UINT32				size						(pANTLR3_INT_STREAM is);


// Helper functions
//
static	void						fillBuffer					(pANTLR3_COMMON_TREE_NODE_STREAM ctns, pANTLR3_BASE_TREE t);
static	void						fillBufferRoot				(pANTLR3_COMMON_TREE_NODE_STREAM ctns);

// Constructors
//
static	void						antlr3TreeNodeStreamFree			(pANTLR3_TREE_NODE_STREAM tns);
static	void						antlr3CommonTreeNodeStreamFree		(pANTLR3_COMMON_TREE_NODE_STREAM ctns);

ANTLR3_API pANTLR3_TREE_NODE_STREAM
antlr3TreeNodeStreamNew()
{
    pANTLR3_TREE_NODE_STREAM stream;

    // Memory for the interface structure
    //
    stream  = (pANTLR3_TREE_NODE_STREAM) ANTLR3_CALLOC(1, sizeof(ANTLR3_TREE_NODE_STREAM));

    if	(stream == NULL)
    {
		return	NULL;
    }

    // Install basic API 
    //
	stream->replaceChildren = replaceChildren;
    stream->free			= antlr3TreeNodeStreamFree;
    
    return stream;
}

static void
antlr3TreeNodeStreamFree(pANTLR3_TREE_NODE_STREAM stream)
{   
    ANTLR3_FREE(stream);
}

ANTLR3_API pANTLR3_COMMON_TREE_NODE_STREAM
antlr3CommonTreeNodeStreamNewTree(pANTLR3_BASE_TREE tree, ANTLR3_UINT32 hint)
{
	pANTLR3_COMMON_TREE_NODE_STREAM stream;

	stream = antlr3CommonTreeNodeStreamNew(tree->strFactory, hint);

	if	(stream == NULL)
	{
		return	NULL;
	}
	stream->root    = tree;

	return stream;
}

ANTLR3_API pANTLR3_COMMON_TREE_NODE_STREAM
antlr3CommonTreeNodeStreamNewStream(pANTLR3_COMMON_TREE_NODE_STREAM inStream)
{
	pANTLR3_COMMON_TREE_NODE_STREAM stream;

	// Memory for the interface structure
	//
	stream  = (pANTLR3_COMMON_TREE_NODE_STREAM) ANTLR3_CALLOC(1, sizeof(ANTLR3_COMMON_TREE_NODE_STREAM));

	if	(stream == NULL)
	{
		return	NULL;
	}

	// Copy in all the reusable parts of the originating stream and create new
	// pieces where necessary.
	//

	// String factory for tree walker
	//
	stream->stringFactory		= inStream->stringFactory;

	// Create an adaptor for the common tree node stream
	//
	stream->adaptor				= inStream->adaptor;

	// Create space for the tree node stream interface
	//
	stream->tnstream	    = antlr3TreeNodeStreamNew();

	if	(stream->tnstream == NULL)
	{
		stream->free				(stream);

		return	NULL;
	}

	// Create space for the INT_STREAM interface
	//
	stream->tnstream->istream		    =  antlr3IntStreamNew();

	if	(stream->tnstream->istream == NULL)
	{
		stream->tnstream->free		(stream->tnstream);
		stream->free				(stream);

		return	NULL;
	}

	// Install the common tree node stream API
	//
	stream->addNavigationNode		    =  addNavigationNode;
	stream->hasUniqueNavigationNodes    =  hasUniqueNavigationNodes;
	stream->newDownNode					=  newDownNode;
	stream->newUpNode					=  newUpNode;
	stream->reset						=  reset;
	stream->push						=  push;
	stream->pop							=  pop;
	stream->getLookaheadSize			=  getLookaheadSize;

	stream->free			    =  antlr3CommonTreeNodeStreamFree;

	// Install the tree node stream API
	//
	stream->tnstream->getTreeAdaptor			=  getTreeAdaptor;
	stream->tnstream->getTreeSource				=  getTreeSource;
	stream->tnstream->_LT						=  _LT;
	stream->tnstream->setUniqueNavigationNodes	=  setUniqueNavigationNodes;
	stream->tnstream->toString					=  toString;
	stream->tnstream->toStringSS				=  toStringSS;
	stream->tnstream->toStringWork				=  toStringWork;
	stream->tnstream->get						=  get;

	// Install INT_STREAM interface
	//
	stream->tnstream->istream->consume	    =  consume;
	stream->tnstream->istream->index	    =  tindex;
	stream->tnstream->istream->_LA			=  _LA;
	stream->tnstream->istream->mark			=  mark;
	stream->tnstream->istream->release	    =  release;
	stream->tnstream->istream->rewind	    =  rewindMark;
	stream->tnstream->istream->rewindLast   =  rewindLast;
	stream->tnstream->istream->seek			=  seek;
	stream->tnstream->istream->size			=  size;

	// Initialize data elements of INT stream
	//
	stream->tnstream->istream->type			= ANTLR3_COMMONTREENODE;
	stream->tnstream->istream->super	    =  (stream->tnstream);

	// Initialize data elements of TREE stream
	//
	stream->tnstream->ctns =  stream;

	// Initialize data elements of the COMMON TREE NODE stream
	//
	stream->super					= NULL;
	stream->uniqueNavigationNodes	= ANTLR3_FALSE;
	stream->markers					= NULL;
	stream->nodeStack				= inStream->nodeStack;

	// Create the node list map
	//
	stream->nodes	= antlr3VectorNew(DEFAULT_INITIAL_BUFFER_SIZE);
	stream->p		= -1;

	// Install the navigation nodes     
	//
	
	// Install the navigation nodes     
	//
	antlr3SetCTAPI(&(stream->UP));
	antlr3SetCTAPI(&(stream->DOWN));
	antlr3SetCTAPI(&(stream->EOF_NODE));
	antlr3SetCTAPI(&(stream->INVALID_NODE));

	stream->UP.token						= inStream->UP.token;
	inStream->UP.token->strFactory			= stream->stringFactory;
	stream->DOWN.token						= inStream->DOWN.token;
	inStream->DOWN.token->strFactory		= stream->stringFactory;
	stream->EOF_NODE.token					= inStream->EOF_NODE.token;
	inStream->EOF_NODE.token->strFactory	= stream->stringFactory;
	stream->INVALID_NODE.token				= inStream->INVALID_NODE.token;
	inStream->INVALID_NODE.token->strFactory= stream->stringFactory;

	// Reuse the root tree of the originating stream
	//
	stream->root		= inStream->root;

	// Signal that this is a rewriting stream so we don't
	// free the originating tree. Anything that we rewrite or
	// duplicate here will be done through the adaptor or 
	// the original tree factory.
	//
	stream->isRewriter	= ANTLR3_TRUE;
	return stream;
}

ANTLR3_API pANTLR3_COMMON_TREE_NODE_STREAM
antlr3CommonTreeNodeStreamNew(pANTLR3_STRING_FACTORY strFactory, ANTLR3_UINT32 hint)
{
	pANTLR3_COMMON_TREE_NODE_STREAM stream;
	pANTLR3_COMMON_TOKEN			token;

	// Memory for the interface structure
	//
	stream  = (pANTLR3_COMMON_TREE_NODE_STREAM) ANTLR3_CALLOC(1, sizeof(ANTLR3_COMMON_TREE_NODE_STREAM));

	if	(stream == NULL)
	{
		return	NULL;
	}

	// String factory for tree walker
	//
	stream->stringFactory		= strFactory;

	// Create an adaptor for the common tree node stream
	//
	stream->adaptor				= ANTLR3_TREE_ADAPTORNew(strFactory);

	if	(stream->adaptor == NULL)
	{
		stream->free(stream);
		return	NULL;
	}

	// Create space for the tree node stream interface
	//
	stream->tnstream	    = antlr3TreeNodeStreamNew();

	if	(stream->tnstream == NULL)
	{
		stream->adaptor->free		(stream->adaptor);
		stream->free				(stream);

		return	NULL;
	}

	// Create space for the INT_STREAM interface
	//
	stream->tnstream->istream		    =  antlr3IntStreamNew();

	if	(stream->tnstream->istream == NULL)
	{
		stream->adaptor->free		(stream->adaptor);
		stream->tnstream->free		(stream->tnstream);
		stream->free				(stream);

		return	NULL;
	}

	// Install the common tree node stream API
	//
	stream->addNavigationNode		    =  addNavigationNode;
	stream->hasUniqueNavigationNodes    =  hasUniqueNavigationNodes;
	stream->newDownNode					=  newDownNode;
	stream->newUpNode					=  newUpNode;
	stream->reset						=  reset;
	stream->push						=  push;
	stream->pop							=  pop;

	stream->free			    =  antlr3CommonTreeNodeStreamFree;

	// Install the tree node stream API
	//
	stream->tnstream->getTreeAdaptor			=  getTreeAdaptor;
	stream->tnstream->getTreeSource				=  getTreeSource;
	stream->tnstream->_LT						=  _LT;
	stream->tnstream->setUniqueNavigationNodes	=  setUniqueNavigationNodes;
	stream->tnstream->toString					=  toString;
	stream->tnstream->toStringSS				=  toStringSS;
	stream->tnstream->toStringWork				=  toStringWork;
	stream->tnstream->get						=  get;

	// Install INT_STREAM interface
	//
	stream->tnstream->istream->consume	    =  consume;
	stream->tnstream->istream->index	    =  tindex;
	stream->tnstream->istream->_LA			=  _LA;
	stream->tnstream->istream->mark			=  mark;
	stream->tnstream->istream->release	    =  release;
	stream->tnstream->istream->rewind	    =  rewindMark;
	stream->tnstream->istream->rewindLast   =  rewindLast;
	stream->tnstream->istream->seek			=  seek;
	stream->tnstream->istream->size			=  size;

	// Initialize data elements of INT stream
	//
	stream->tnstream->istream->type			= ANTLR3_COMMONTREENODE;
	stream->tnstream->istream->super	    =  (stream->tnstream);

	// Initialize data elements of TREE stream
	//
	stream->tnstream->ctns =  stream;

	// Initialize data elements of the COMMON TREE NODE stream
	//
	stream->super					= NULL;
	stream->uniqueNavigationNodes	= ANTLR3_FALSE;
	stream->markers					= NULL;
	stream->nodeStack				= antlr3StackNew(INITIAL_CALL_STACK_SIZE);

	// Create the node list map
	//
	if	(hint == 0)
	{
		hint = DEFAULT_INITIAL_BUFFER_SIZE;
	}
	stream->nodes	= antlr3VectorNew(hint);
	stream->p		= -1;

	// Install the navigation nodes     
	//
	antlr3SetCTAPI(&(stream->UP));
	antlr3SetCTAPI(&(stream->DOWN));
	antlr3SetCTAPI(&(stream->EOF_NODE));
	antlr3SetCTAPI(&(stream->INVALID_NODE));

	token						= antlr3CommonTokenNew(ANTLR3_TOKEN_UP);
	token->strFactory			= strFactory;
	token->textState			= ANTLR3_TEXT_CHARP;
	token->tokText.chars		= (pANTLR3_UCHAR)"UP";
	stream->UP.token			= token;

	token						= antlr3CommonTokenNew(ANTLR3_TOKEN_DOWN);
	token->strFactory			= strFactory;
	token->textState			= ANTLR3_TEXT_CHARP;
	token->tokText.chars		= (pANTLR3_UCHAR)"DOWN";
	stream->DOWN.token			= token;

	token						= antlr3CommonTokenNew(ANTLR3_TOKEN_EOF);
	token->strFactory			= strFactory;
	token->textState			= ANTLR3_TEXT_CHARP;
	token->tokText.chars		= (pANTLR3_UCHAR)"EOF";
	stream->EOF_NODE.token		= token;

	token						= antlr3CommonTokenNew(ANTLR3_TOKEN_INVALID);
	token->strFactory			= strFactory;
	token->textState			= ANTLR3_TEXT_CHARP;
	token->tokText.chars		= (pANTLR3_UCHAR)"INVALID";
	stream->INVALID_NODE.token	= token;


	return  stream;
}

/// Free up any resources that belong to this common tree node stream.
///
static	void			    antlr3CommonTreeNodeStreamFree  (pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{

	// If this is a rewrting stream, then certain resources
	// belong to the originating node stream and we do not
	// free them here.
	//
	if	(ctns->isRewriter != ANTLR3_TRUE)
	{
		ctns->adaptor			->free  (ctns->adaptor);

		if	(ctns->nodeStack != NULL)
		{
			ctns->nodeStack->free(ctns->nodeStack);
		}

		ANTLR3_FREE(ctns->INVALID_NODE.token);
		ANTLR3_FREE(ctns->EOF_NODE.token);
		ANTLR3_FREE(ctns->DOWN.token);
		ANTLR3_FREE(ctns->UP.token);
	}
	
	if	(ctns->nodes != NULL)
	{
		ctns->nodes			->free  (ctns->nodes);
	}
	ctns->tnstream->istream ->free  (ctns->tnstream->istream);
    ctns->tnstream			->free  (ctns->tnstream);


    ANTLR3_FREE(ctns);
}

// ------------------------------------------------------------------------------
// Local helpers
//

/// Walk and fill the tree node buffer from the root tree
///
static void
fillBufferRoot(pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{
	// Call the generic buffer routine with the root as the
	// argument
	//
	fillBuffer(ctns, ctns->root);
	ctns->p = 0;					// Indicate we are at buffer start
}

/// Walk tree with depth-first-search and fill nodes buffer.
/// Don't add in DOWN, UP nodes if the supplied tree is a list (t is isNilNode)
// such as the root tree is.
///
static void
fillBuffer(pANTLR3_COMMON_TREE_NODE_STREAM ctns, pANTLR3_BASE_TREE t)
{
	ANTLR3_BOOLEAN	nilNode;
	ANTLR3_UINT32	nCount;
	ANTLR3_UINT32	c;

	nilNode = ctns->adaptor->isNilNode(ctns->adaptor, t);

	// If the supplied node is not a nil (list) node then we
	// add in the node itself to the vector
	//
	if	(nilNode == ANTLR3_FALSE)
	{
		ctns->nodes->add(ctns->nodes, t, NULL);	
	}

	// Only add a DOWN node if the tree is not a nil tree and
	// the tree does have children.
	//
	nCount = t->getChildCount(t);

	if	(nilNode == ANTLR3_FALSE && nCount>0)
	{
		ctns->addNavigationNode(ctns, ANTLR3_TOKEN_DOWN);
	}

	// We always add any children the tree contains, which is
	// a recursive call to this function, which will cause similar
	// recursion and implement a depth first addition
	//
	for	(c = 0; c < nCount; c++)
	{
		fillBuffer(ctns, ctns->adaptor->getChild(ctns->adaptor, t, c));
	}

	// If the tree had children and was not a nil (list) node, then we
	// we need to add an UP node here to match the DOWN node
	//
	if	(nilNode == ANTLR3_FALSE && nCount > 0)
	{
		ctns->addNavigationNode(ctns, ANTLR3_TOKEN_UP);
	}
}


// ------------------------------------------------------------------------------
// Interface functions
//

/// Reset the input stream to the start of the input nodes.
///
static	void		
reset	    (pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{
	if	(ctns->p != -1)
	{
		ctns->p									= 0;
	}
	ctns->tnstream->istream->lastMarker		= 0;


	// Free and reset the node stack only if this is not
	// a rewriter, which is going to reuse the originating
	// node streams node stack
	//
	if  (ctns->isRewriter != ANTLR3_TRUE)
    {
		if	(ctns->nodeStack != NULL)
		{
			ctns->nodeStack->free(ctns->nodeStack);
			ctns->nodeStack = antlr3StackNew(INITIAL_CALL_STACK_SIZE);
		}
	}
}


static pANTLR3_BASE_TREE
LB(pANTLR3_TREE_NODE_STREAM tns, ANTLR3_INT32 k)
{
	if	( k==0)
	{
		return	&(tns->ctns->INVALID_NODE.baseTree);
	}

	if	( (tns->ctns->p - k) < 0)
	{
		return	&(tns->ctns->INVALID_NODE.baseTree);
	}

	return tns->ctns->nodes->get(tns->ctns->nodes, tns->ctns->p - k);
}

/// Get tree node at current input pointer + i ahead where i=1 is next node.
/// i<0 indicates nodes in the past.  So -1 is previous node and -2 is
/// two nodes ago. LT(0) is undefined.  For i>=n, return null.
/// Return null for LT(0) and any index that results in an absolute address
/// that is negative.
///
/// This is analogous to the _LT() method of the TokenStream, but this
/// returns a tree node instead of a token.  Makes code gen identical
/// for both parser and tree grammars. :)
///
static	pANTLR3_BASE_TREE	    
_LT	    (pANTLR3_TREE_NODE_STREAM tns, ANTLR3_INT32 k)
{
	if	(tns->ctns->p == -1)
	{
		fillBufferRoot(tns->ctns);
	}

	if	(k < 0)
	{
		return LB(tns, -k);
	}
	else if	(k == 0)
	{
		return	&(tns->ctns->INVALID_NODE.baseTree);
	}

	// k was a legitimate request, 
	//
	if	(( tns->ctns->p + k - 1) >= (ANTLR3_INT32)(tns->ctns->nodes->count))
	{
		return &(tns->ctns->EOF_NODE.baseTree);
	}

	return	tns->ctns->nodes->get(tns->ctns->nodes, tns->ctns->p + k - 1);
}

/// Where is this stream pulling nodes from?  This is not the name, but
/// the object that provides node objects.
///
static	pANTLR3_BASE_TREE	    
getTreeSource	(pANTLR3_TREE_NODE_STREAM tns)
{
    return  tns->ctns->root;
}

/// Consume the next node from the input stream
///
static	void		    
consume	(pANTLR3_INT_STREAM is)
{
    pANTLR3_TREE_NODE_STREAM		tns;
    pANTLR3_COMMON_TREE_NODE_STREAM	ctns;

    tns	    = (pANTLR3_TREE_NODE_STREAM)(is->super);
    ctns    = tns->ctns;

	if	(ctns->p == -1)
	{
		fillBufferRoot(ctns);
	}
	ctns->p++;
}

static	ANTLR3_UINT32	    
_LA	    (pANTLR3_INT_STREAM is, ANTLR3_INT32 i)
{
	pANTLR3_TREE_NODE_STREAM		tns;
	pANTLR3_BASE_TREE				t;

	tns	    = (pANTLR3_TREE_NODE_STREAM)(is->super);

	// Ask LT for the 'token' at that position
	//
	t = tns->_LT(tns, i);

	if	(t == NULL)
	{
		return	ANTLR3_TOKEN_INVALID;
	}

	// Token node was there so return the type of it
	//
	return  t->getType(t);
}

/// Mark the state of the input stream so that we can come back to it
/// after a syntactic predicate and so on.
///
static	ANTLR3_MARKER	    
mark	(pANTLR3_INT_STREAM is)
{
	pANTLR3_TREE_NODE_STREAM		tns;
	pANTLR3_COMMON_TREE_NODE_STREAM	ctns;

	tns	    = (pANTLR3_TREE_NODE_STREAM)(is->super);
	ctns    = tns->ctns;

	if	(tns->ctns->p == -1)
	{
		fillBufferRoot(tns->ctns);
	}

	// Return the current mark point
	//
	ctns->tnstream->istream->lastMarker = ctns->tnstream->istream->index(ctns->tnstream->istream);

	return ctns->tnstream->istream->lastMarker;
}

static	void		    
release	(pANTLR3_INT_STREAM is, ANTLR3_MARKER marker)
{
}

/// Rewind the current state of the tree walk to the state it
/// was in when mark() was called and it returned marker.  Also,
/// wipe out the lookahead which will force reloading a few nodes
/// but it is better than making a copy of the lookahead buffer
/// upon mark().
///
static	void		    
rewindMark	    (pANTLR3_INT_STREAM is, ANTLR3_MARKER marker)
{
	is->seek(is, marker);
}

static	void		    
rewindLast	(pANTLR3_INT_STREAM is)
{
   is->seek(is, is->lastMarker);
}

/// consume() ahead until we hit index.  Can't just jump ahead--must
/// spit out the navigation nodes.
///
static	void		    
seek	(pANTLR3_INT_STREAM is, ANTLR3_MARKER index)
{
    pANTLR3_TREE_NODE_STREAM		tns;
    pANTLR3_COMMON_TREE_NODE_STREAM	ctns;

    tns	    = (pANTLR3_TREE_NODE_STREAM)(is->super);
    ctns    = tns->ctns;

	ctns->p = ANTLR3_UINT32_CAST(index);
}

static	ANTLR3_MARKER		    
tindex	(pANTLR3_INT_STREAM is)
{
    pANTLR3_TREE_NODE_STREAM		tns;
    pANTLR3_COMMON_TREE_NODE_STREAM	ctns;

    tns	    = (pANTLR3_TREE_NODE_STREAM)(is->super);
    ctns    = tns->ctns;

	return (ANTLR3_MARKER)(ctns->p);
}

/// Expensive to compute the size of the whole tree while parsing.
/// This method only returns how much input has been seen so far.  So
/// after parsing it returns true size.
///
static	ANTLR3_UINT32		    
size	(pANTLR3_INT_STREAM is)
{
    pANTLR3_TREE_NODE_STREAM		tns;
    pANTLR3_COMMON_TREE_NODE_STREAM	ctns;

    tns	    = (pANTLR3_TREE_NODE_STREAM)(is->super);
    ctns    = tns->ctns;

	if	(ctns->p == -1)
	{
		fillBufferRoot(ctns);
	}

	return ctns->nodes->size(ctns->nodes);
}

/// As we flatten the tree, we use UP, DOWN nodes to represent
/// the tree structure.  When debugging we need unique nodes
/// so instantiate new ones when uniqueNavigationNodes is true.
///
static	void		    
addNavigationNode	    (pANTLR3_COMMON_TREE_NODE_STREAM ctns, ANTLR3_UINT32 ttype)
{
	pANTLR3_BASE_TREE	    node;

	node = NULL;

	if	(ttype == ANTLR3_TOKEN_DOWN)
	{
		if  (ctns->hasUniqueNavigationNodes(ctns) == ANTLR3_TRUE)
		{
			node    = ctns->newDownNode(ctns);
		}
		else
		{
			node    = &(ctns->DOWN.baseTree);
		}
	}
	else
	{
		if  (ctns->hasUniqueNavigationNodes(ctns) == ANTLR3_TRUE)
		{
			node    = ctns->newUpNode(ctns);
		}
		else
		{
			node    = &(ctns->UP.baseTree);
		}
	}

	// Now add the node we decided upon.
	//
	ctns->nodes->add(ctns->nodes, node, NULL);
}


static	pANTLR3_BASE_TREE_ADAPTOR			    
getTreeAdaptor	(pANTLR3_TREE_NODE_STREAM tns)
{
    return  tns->ctns->adaptor;
}

static	ANTLR3_BOOLEAN	    
hasUniqueNavigationNodes	    (pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{
    return  ctns->uniqueNavigationNodes;
}

static	void		    
setUniqueNavigationNodes	    (pANTLR3_TREE_NODE_STREAM tns, ANTLR3_BOOLEAN uniqueNavigationNodes)
{
    tns->ctns->uniqueNavigationNodes = uniqueNavigationNodes;
}


/// Print out the entire tree including DOWN/UP nodes.  Uses
/// a recursive walk.  Mostly useful for testing as it yields
/// the token types not text.
///
static	pANTLR3_STRING	    
toString	    (pANTLR3_TREE_NODE_STREAM tns)
{

    return  tns->toStringSS(tns, tns->ctns->root, NULL);
}

static	pANTLR3_STRING	    
toStringSS	    (pANTLR3_TREE_NODE_STREAM tns, pANTLR3_BASE_TREE start, pANTLR3_BASE_TREE stop)
{
    pANTLR3_STRING  buf;

    buf = tns->ctns->stringFactory->newRaw(tns->ctns->stringFactory);

    tns->toStringWork(tns, start, stop, buf);

    return  buf;
}

static	void	    
toStringWork	(pANTLR3_TREE_NODE_STREAM tns, pANTLR3_BASE_TREE p, pANTLR3_BASE_TREE stop, pANTLR3_STRING buf)
{

	ANTLR3_UINT32   n;
	ANTLR3_UINT32   c;

	if	(!p->isNilNode(p) )
	{
		pANTLR3_STRING	text;

		text	= p->toString(p);

		if  (text == NULL)
		{
			text = tns->ctns->stringFactory->newRaw(tns->ctns->stringFactory);

			text->addc	(text, ' ');
			text->addi	(text, p->getType(p));
		}

		buf->appendS(buf, text);
	}

	if	(p == stop)
	{
		return;		/* Finished */
	}

	n = p->getChildCount(p);

	if	(n > 0 && ! p->isNilNode(p) )
	{
		buf->addc   (buf, ' ');
		buf->addi   (buf, ANTLR3_TOKEN_DOWN);
	}

	for	(c = 0; c<n ; c++)
	{
		pANTLR3_BASE_TREE   child;

		child = p->getChild(p, c);
		tns->toStringWork(tns, child, stop, buf);
	}

	if	(n > 0 && ! p->isNilNode(p) )
	{
		buf->addc   (buf, ' ');
		buf->addi   (buf, ANTLR3_TOKEN_UP);
	}
}

static	ANTLR3_UINT32	    
getLookaheadSize	(pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{
    return	ctns->tail < ctns->head 
	    ?	(ctns->lookAheadLength - ctns->head + ctns->tail)
	    :	(ctns->tail - ctns->head);
}

static	pANTLR3_BASE_TREE	    
newDownNode		(pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{
    pANTLR3_COMMON_TREE	    dNode;
    pANTLR3_COMMON_TOKEN    token;

    token					= antlr3CommonTokenNew(ANTLR3_TOKEN_DOWN);
	token->textState		= ANTLR3_TEXT_CHARP;
	token->tokText.chars	= (pANTLR3_UCHAR)"DOWN";
    dNode					= antlr3CommonTreeNewFromToken(token);

    return  &(dNode->baseTree);
}

static	pANTLR3_BASE_TREE	    
newUpNode		(pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{
    pANTLR3_COMMON_TREE	    uNode;
    pANTLR3_COMMON_TOKEN    token;

    token					= antlr3CommonTokenNew(ANTLR3_TOKEN_UP);
	token->textState		= ANTLR3_TEXT_CHARP;
	token->tokText.chars	= (pANTLR3_UCHAR)"UP";
    uNode					= antlr3CommonTreeNewFromToken(token);

    return  &(uNode->baseTree);
}

/// Replace from start to stop child index of parent with t, which might
/// be a list.  Number of children may be different
/// after this call.  The stream is notified because it is walking the
/// tree and might need to know you are monkey-ing with the underlying
/// tree.  Also, it might be able to modify the node stream to avoid
/// re-streaming for future phases.
///
/// If parent is null, don't do anything; must be at root of overall tree.
/// Can't replace whatever points to the parent externally.  Do nothing.
///
static	void						
replaceChildren				(pANTLR3_TREE_NODE_STREAM tns, pANTLR3_BASE_TREE parent, ANTLR3_INT32 startChildIndex, ANTLR3_INT32 stopChildIndex, pANTLR3_BASE_TREE t)
{
	if	(parent != NULL)
	{
		pANTLR3_BASE_TREE_ADAPTOR	adaptor;
		pANTLR3_COMMON_TREE_ADAPTOR	cta;

		adaptor	= tns->getTreeAdaptor(tns);
		cta		= (pANTLR3_COMMON_TREE_ADAPTOR)(adaptor->super);

		adaptor->replaceChildren(adaptor, parent, startChildIndex, stopChildIndex, t);
	}
}

static	pANTLR3_BASE_TREE
get							(pANTLR3_TREE_NODE_STREAM tns, ANTLR3_INT32 k)
{
	if	(tns->ctns->p == -1)
	{
		fillBufferRoot(tns->ctns);
	}

	return tns->ctns->nodes->get(tns->ctns->nodes, k);
}

static	void
push						(pANTLR3_COMMON_TREE_NODE_STREAM ctns, ANTLR3_INT32 index)
{
	ctns->nodeStack->push(ctns->nodeStack, ANTLR3_FUNC_PTR(ctns->p), NULL);	// Save current index
	ctns->tnstream->istream->seek(ctns->tnstream->istream, index);
}

static	ANTLR3_INT32
pop							(pANTLR3_COMMON_TREE_NODE_STREAM ctns)
{
	ANTLR3_INT32	retVal;

	retVal = ANTLR3_UINT32_CAST(ctns->nodeStack->pop(ctns->nodeStack));
	ctns->tnstream->istream->seek(ctns->tnstream->istream, retVal);
	return retVal;
}
