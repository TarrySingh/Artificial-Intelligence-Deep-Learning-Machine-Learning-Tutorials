/** \file
 *  Implementation of the tree parser and overrides for the base recognizer
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

#include    <antlr3treeparser.h>

/* BASE Recognizer overrides
 */
static void				mismatch	    (pANTLR3_BASE_RECOGNIZER recognizer, ANTLR3_UINT32 ttype, pANTLR3_BITSET_LIST follow);

/* Tree parser API
 */
static void			setTreeNodeStream	    (pANTLR3_TREE_PARSER parser, pANTLR3_COMMON_TREE_NODE_STREAM input);
static pANTLR3_COMMON_TREE_NODE_STREAM	
					getTreeNodeStream	    (pANTLR3_TREE_PARSER parser);
static void			freeParser				(pANTLR3_TREE_PARSER parser);    
static void *		getCurrentInputSymbol	(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM istream);
static void *		getMissingSymbol		(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM	istream, pANTLR3_EXCEPTION	e,
												ANTLR3_UINT32 expectedTokenType, pANTLR3_BITSET_LIST follow);


ANTLR3_API pANTLR3_TREE_PARSER
antlr3TreeParserNewStream(ANTLR3_UINT32 sizeHint, pANTLR3_COMMON_TREE_NODE_STREAM ctnstream, pANTLR3_RECOGNIZER_SHARED_STATE state)
{
	pANTLR3_TREE_PARSER	    parser;

	/** Allocate tree parser memory
	*/
	parser  =(pANTLR3_TREE_PARSER) ANTLR3_MALLOC(sizeof(ANTLR3_TREE_PARSER));

	if	(parser == NULL)
	{
		return	NULL;
	}

	/* Create and install a base recognizer which does most of the work for us
	*/
	parser->rec =  antlr3BaseRecognizerNew(ANTLR3_TYPE_PARSER, sizeHint, state);

	if	(parser->rec == NULL)
	{
		parser->free(parser);
		return	NULL;
	}

	/* Ensure we can track back to the tree parser super structure
	* from the base recognizer structure
	*/
	parser->rec->super	= parser;
	parser->rec->type	= ANTLR3_TYPE_TREE_PARSER;

	/* Install our base recognizer overrides
	*/
	parser->rec->mismatch				= mismatch;
	parser->rec->exConstruct			= antlr3MTNExceptionNew;
	parser->rec->getCurrentInputSymbol	= getCurrentInputSymbol;
	parser->rec->getMissingSymbol		= getMissingSymbol;

	/* Install tree parser API
	*/
	parser->getTreeNodeStream	=  getTreeNodeStream;
	parser->setTreeNodeStream	=  setTreeNodeStream;
	parser->free		=  freeParser;

	/* Install the tree node stream
	*/
	parser->setTreeNodeStream(parser, ctnstream);

	return  parser;
}

/**
 * \brief
 * Creates a new Mismatched Tree Nde Exception and inserts in the recognizer
 * exception stack.
 * 
 * \param recognizer
 * Context pointer for this recognizer
 * 
 */
ANTLR3_API	void
antlr3MTNExceptionNew(pANTLR3_BASE_RECOGNIZER recognizer)
{
    /* Create a basic recognition exception structure
     */
    antlr3RecognitionExceptionNew(recognizer);

    /* Now update it to indicate this is a Mismatched token exception
     */
    recognizer->state->exception->name		= ANTLR3_MISMATCHED_TREE_NODE_NAME;
    recognizer->state->exception->type		= ANTLR3_MISMATCHED_TREE_NODE_EXCEPTION;

    return;
}


static void
freeParser	(pANTLR3_TREE_PARSER parser)
{
	if	(parser->rec != NULL)
	{
		// This may have ben a delegate or delegator parser, in which case the
		// state may already have been freed (and set to NULL therefore)
		// so we ignore the state if we don't have it.
		//
		if	(parser->rec->state != NULL)
		{
			if	(parser->rec->state->following != NULL)
			{
				parser->rec->state->following->free(parser->rec->state->following);
				parser->rec->state->following = NULL;
			}
		}
	    parser->rec->free(parser->rec);
	    parser->rec	= NULL;
    }

    ANTLR3_FREE(parser);
}

/** Set the input stream and reset the parser
 */
static void
setTreeNodeStream	(pANTLR3_TREE_PARSER parser, pANTLR3_COMMON_TREE_NODE_STREAM input)
{
    parser->ctnstream = input;
    parser->rec->reset		(parser->rec);
    parser->ctnstream->reset	(parser->ctnstream);
}

/** Return a pointer to the input stream
 */
static pANTLR3_COMMON_TREE_NODE_STREAM
getTreeNodeStream	(pANTLR3_TREE_PARSER parser)
{
    return  parser->ctnstream;
}


/** Override for standard base recognizer mismatch function
 *  as we have DOWN/UP nodes in the stream that have no line info,
 *  plus we want to alter the exception type.
 */
static void
mismatch	    (pANTLR3_BASE_RECOGNIZER recognizer, ANTLR3_UINT32 ttype, pANTLR3_BITSET_LIST follow)
{
    recognizer->exConstruct(recognizer);
    recognizer->recoverFromMismatchedToken(recognizer, ttype, follow);
}

#ifdef ANTLR3_WINDOWS
#pragma warning	(push)
#pragma warning (disable : 4100)
#endif

// Default implementation is for parser and assumes a token stream as supplied by the runtime.
// You MAY need override this function if the standard TOKEN_STREAM is not what you are using.
//
static void *				
getCurrentInputSymbol		(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM istream)
{
	pANTLR3_TREE_NODE_STREAM		tns;
    pANTLR3_COMMON_TREE_NODE_STREAM	ctns;

    tns	    = (pANTLR3_TREE_NODE_STREAM)(istream->super);
    ctns    = tns->ctns;
	return tns->_LT(tns, 1);
}


// Default implementation is for parser and assumes a token stream as supplied by the runtime.
// You MAY need override this function if the standard BASE_TREE is not what you are using.
//
static void *				
getMissingSymbol			(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM	istream, pANTLR3_EXCEPTION	e,
									ANTLR3_UINT32 expectedTokenType, pANTLR3_BITSET_LIST follow)
{
	pANTLR3_TREE_NODE_STREAM		tns;
    pANTLR3_COMMON_TREE_NODE_STREAM	ctns;
	pANTLR3_BASE_TREE				node;
	pANTLR3_BASE_TREE				current;
	pANTLR3_COMMON_TOKEN			token;
	pANTLR3_STRING					text;

	// Dereference the standard pointers
	//
    tns	    = (pANTLR3_TREE_NODE_STREAM)(istream->super);
    ctns    = tns->ctns;

	// Create a new empty node, by stealing the current one, or the previous one if the current one is EOF
	//
	current	= tns->_LT(tns, 1);

	if	(current == &ctns->EOF_NODE.baseTree)
	{
		current = tns->_LT(tns, -1);
	}
	node	= current->dupNode(current);

	// Find the newly dupicated token
	//
	token	= node->getToken(node);

	// Create the token text that shows it has been inserted
	//
	token->setText8			(token, (pANTLR3_UINT8)"<missing ");
	text = token->getText	(token);
	text->append8			(text, (const char *)recognizer->state->tokenNames[expectedTokenType]);
	text->append8			(text, (const char *)">");
	
	// Finally return the pointer to our new node
	//
	return	node;
}
#ifdef ANTLR3_WINDOWS
#pragma warning	(pop)
#endif

