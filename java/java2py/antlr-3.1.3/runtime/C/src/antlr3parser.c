/** \file
 * Implementation of the base functionality for an ANTLR3 parser.
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

#include    <antlr3parser.h>

/* Parser API 
 */
static void					setDebugListener	(pANTLR3_PARSER parser, pANTLR3_DEBUG_EVENT_LISTENER dbg);
static void					setTokenStream		(pANTLR3_PARSER parser, pANTLR3_TOKEN_STREAM);
static pANTLR3_TOKEN_STREAM	getTokenStream		(pANTLR3_PARSER parser);
static void					freeParser		    (pANTLR3_PARSER parser);

ANTLR3_API pANTLR3_PARSER
antlr3ParserNewStreamDbg		(ANTLR3_UINT32 sizeHint, pANTLR3_TOKEN_STREAM tstream, pANTLR3_DEBUG_EVENT_LISTENER dbg, pANTLR3_RECOGNIZER_SHARED_STATE state)
{
	pANTLR3_PARSER	parser;

	parser = antlr3ParserNewStream(sizeHint, tstream, state);

	if	(parser == NULL)
    {
		return	NULL;
    }

	parser->setDebugListener(parser, dbg);

	return parser;
}

ANTLR3_API pANTLR3_PARSER
antlr3ParserNew		(ANTLR3_UINT32 sizeHint, pANTLR3_RECOGNIZER_SHARED_STATE state)
{
    pANTLR3_PARSER	parser;

    /* Allocate memory
     */
    parser	= (pANTLR3_PARSER) ANTLR3_MALLOC(sizeof(ANTLR3_PARSER));

    if	(parser == NULL)
    {
		return	NULL;
    }

    /* Install a base parser
     */
    parser->rec =  antlr3BaseRecognizerNew(ANTLR3_TYPE_PARSER, sizeHint, state);

    if	(parser->rec == NULL)
    {
		parser->free(parser);
		return	NULL;
    }

    parser->rec->super	= parser;

    /* Parser overrides
     */
    parser->rec->exConstruct	=  antlr3MTExceptionNew;

    /* Install the API
     */
	parser->setDebugListener	=  setDebugListener;
    parser->setTokenStream		=  setTokenStream;
    parser->getTokenStream		=  getTokenStream;

    parser->free			=  freeParser;

    return parser;
}

ANTLR3_API pANTLR3_PARSER
antlr3ParserNewStream	(ANTLR3_UINT32 sizeHint, pANTLR3_TOKEN_STREAM tstream, pANTLR3_RECOGNIZER_SHARED_STATE state)
{
    pANTLR3_PARSER	parser;

    parser  = antlr3ParserNew(sizeHint, state);

    if	(parser == NULL)
    {
		return	NULL;
    }

    /* Everything seems to be hunky dory so we can install the 
     * token stream.
     */
    parser->setTokenStream(parser, tstream);

    return parser;
}

static void		
freeParser			    (pANTLR3_PARSER parser)
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

static void					
setDebugListener		(pANTLR3_PARSER parser, pANTLR3_DEBUG_EVENT_LISTENER dbg)
{
	// Set the debug listener. There are no methods to override
	// because currently the only ones that notify the debugger
	// are error reporting and recovery. Hence we can afford to
	// check and see if the debugger interface is null or not
	// there. If there is ever an occasion for a performance
	// sensitive function to use the debugger interface, then
	// a replacement function for debug mode should be supplied
	// and installed here.
	//
	parser->rec->debugger	= dbg;

	// If there was a tokenstream installed already
	// then we need to tell it about the debug interface
	//
	if	(parser->tstream != NULL)
	{
		parser->tstream->setDebugListener(parser->tstream, dbg);
	}
}

static void			
setTokenStream		    (pANTLR3_PARSER parser, pANTLR3_TOKEN_STREAM tstream)
{
    parser->tstream = tstream;
    parser->rec->reset(parser->rec);
}

static pANTLR3_TOKEN_STREAM	
getTokenStream		    (pANTLR3_PARSER parser)
{
    return  parser->tstream;
}














