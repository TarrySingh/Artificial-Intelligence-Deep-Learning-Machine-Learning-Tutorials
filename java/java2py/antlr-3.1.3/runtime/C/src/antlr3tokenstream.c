/// \file 
/// Default implementation of CommonTokenStream
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

#include    <antlr3tokenstream.h>

#ifdef	ANTLR3_WINDOWS
#pragma warning( disable : 4100 )
#endif

// COMMON_TOKEN_STREAM API
//
static void					setTokenTypeChannel	(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_UINT32 ttype, ANTLR3_UINT32 channel);
static void					discardTokenType	(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_INT32 ttype);
static void					discardOffChannel	(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_BOOLEAN discard);
static pANTLR3_VECTOR		getTokens			(pANTLR3_COMMON_TOKEN_STREAM cts);
static pANTLR3_LIST			getTokenRange		(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_UINT32 start, ANTLR3_UINT32 stop);
static pANTLR3_LIST			getTokensSet		(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_UINT32 start, ANTLR3_UINT32 stop, pANTLR3_BITSET types);
static pANTLR3_LIST			getTokensList		(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_UINT32 start, ANTLR3_UINT32 stop, pANTLR3_LIST list);
static pANTLR3_LIST			getTokensType		(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_UINT32 start, ANTLR3_UINT32 stop, ANTLR3_UINT32 type);

// TOKEN_STREAM API 
//
static pANTLR3_COMMON_TOKEN tokLT				(pANTLR3_TOKEN_STREAM ts, ANTLR3_INT32 k);
static pANTLR3_COMMON_TOKEN dbgTokLT			(pANTLR3_TOKEN_STREAM ts, ANTLR3_INT32 k);
static pANTLR3_COMMON_TOKEN get					(pANTLR3_TOKEN_STREAM ts, ANTLR3_UINT32 i);
static pANTLR3_TOKEN_SOURCE getTokenSource		(pANTLR3_TOKEN_STREAM ts);
static void					setTokenSource		(pANTLR3_TOKEN_STREAM ts, pANTLR3_TOKEN_SOURCE tokenSource);
static pANTLR3_STRING	    toString			(pANTLR3_TOKEN_STREAM ts);
static pANTLR3_STRING	    toStringSS			(pANTLR3_TOKEN_STREAM ts, ANTLR3_UINT32 start, ANTLR3_UINT32 stop);
static pANTLR3_STRING	    toStringTT			(pANTLR3_TOKEN_STREAM ts, pANTLR3_COMMON_TOKEN start, pANTLR3_COMMON_TOKEN stop);
static void					setDebugListener	(pANTLR3_TOKEN_STREAM ts, pANTLR3_DEBUG_EVENT_LISTENER debugger);

// INT STREAM API
//
static void					consume						(pANTLR3_INT_STREAM is);
static void					dbgConsume					(pANTLR3_INT_STREAM is);
static ANTLR3_UINT32	    _LA							(pANTLR3_INT_STREAM is, ANTLR3_INT32 i);
static ANTLR3_UINT32	    dbgLA						(pANTLR3_INT_STREAM is, ANTLR3_INT32 i);
static ANTLR3_MARKER	    mark						(pANTLR3_INT_STREAM is);
static ANTLR3_MARKER	    dbgMark						(pANTLR3_INT_STREAM is);
static void					release						(pANTLR3_INT_STREAM is, ANTLR3_MARKER mark);
static ANTLR3_UINT32	    size						(pANTLR3_INT_STREAM is);
static ANTLR3_MARKER		tindex						(pANTLR3_INT_STREAM is);
static void					rewindStream				(pANTLR3_INT_STREAM is, ANTLR3_MARKER marker);
static void					dbgRewindStream				(pANTLR3_INT_STREAM is, ANTLR3_MARKER marker);
static void					rewindLast					(pANTLR3_INT_STREAM is);
static void					dbgRewindLast				(pANTLR3_INT_STREAM is);
static void					seek						(pANTLR3_INT_STREAM is, ANTLR3_MARKER index);
static void					dbgSeek						(pANTLR3_INT_STREAM is, ANTLR3_MARKER index);
static pANTLR3_STRING		getSourceName				(pANTLR3_INT_STREAM is);
static void					antlr3TokenStreamFree		(pANTLR3_TOKEN_STREAM	    stream);
static void					antlr3CTSFree				(pANTLR3_COMMON_TOKEN_STREAM    stream);

// Helpers
//
static void					fillBuffer					(pANTLR3_COMMON_TOKEN_STREAM tokenStream);
static ANTLR3_UINT32	    skipOffTokenChannels		(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_INT32 i);
static ANTLR3_UINT32	    skipOffTokenChannelsReverse	(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_INT32 i);
static pANTLR3_COMMON_TOKEN LB							(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_INT32 i);

ANTLR3_API pANTLR3_TOKEN_STREAM
antlr3TokenStreamNew()
{
    pANTLR3_TOKEN_STREAM stream;

    // Memory for the interface structure
    //
    stream  = (pANTLR3_TOKEN_STREAM) ANTLR3_MALLOC(sizeof(ANTLR3_TOKEN_STREAM));

    if	(stream == NULL)
    {
		return	NULL;
    }

    // Install basic API 
    //
    stream->free    =  antlr3TokenStreamFree;

    
    return stream;
}

static void
antlr3TokenStreamFree(pANTLR3_TOKEN_STREAM stream)
{   
    ANTLR3_FREE(stream);
}

static void		    
antlr3CTSFree	    (pANTLR3_COMMON_TOKEN_STREAM stream)
{
	// We only free up our subordinate interfaces if they belong
	// to us, otherwise we let whoever owns them deal with them.
	//
	if	(stream->tstream->super == stream)
	{
		if	(stream->tstream->istream->super == stream->tstream)
		{
			stream->tstream->istream->free(stream->tstream->istream);
			stream->tstream->istream = NULL;
		}
		stream->tstream->free(stream->tstream);
	}

	// Now we free our own resources
	//
	if	(stream->tokens != NULL)
	{
		stream->tokens->free(stream->tokens);
		stream->tokens	= NULL;
	}
	if	(stream->discardSet != NULL)
	{
		stream->discardSet->free(stream->discardSet);
		stream->discardSet  = NULL;
	}
	if	(stream->channelOverrides != NULL)
	{
		stream->channelOverrides->free(stream->channelOverrides);
		stream->channelOverrides = NULL;
	}

	// Free our memory now
	//
	ANTLR3_FREE(stream);
}

ANTLR3_API pANTLR3_COMMON_TOKEN_STREAM
antlr3CommonTokenDebugStreamSourceNew(ANTLR3_UINT32 hint, pANTLR3_TOKEN_SOURCE source, pANTLR3_DEBUG_EVENT_LISTENER debugger)
{
    pANTLR3_COMMON_TOKEN_STREAM	stream;

	// Create a standard token stream
	//
	stream = antlr3CommonTokenStreamSourceNew(hint, source);

	// Install the debugger object
	//
	stream->tstream->debugger = debugger;

	// Override standard token stream methods with debugging versions
	//
	stream->tstream->initialStreamState	= ANTLR3_FALSE;

	stream->tstream->_LT				= dbgTokLT;

	stream->tstream->istream->consume		= dbgConsume;
	stream->tstream->istream->_LA			= dbgLA;
	stream->tstream->istream->mark			= dbgMark;
	stream->tstream->istream->rewind		= dbgRewindStream;
	stream->tstream->istream->rewindLast	= dbgRewindLast;
	stream->tstream->istream->seek			= dbgSeek;

	return stream;
}

ANTLR3_API pANTLR3_COMMON_TOKEN_STREAM
antlr3CommonTokenStreamSourceNew(ANTLR3_UINT32 hint, pANTLR3_TOKEN_SOURCE source)
{
    pANTLR3_COMMON_TOKEN_STREAM	stream;

    stream = antlr3CommonTokenStreamNew(hint);

    stream->channel = ANTLR3_TOKEN_DEFAULT_CHANNEL;
    
    stream->channelOverrides	= NULL;
    stream->discardSet		= NULL;
    stream->discardOffChannel	= ANTLR3_FALSE;

    stream->tstream->setTokenSource(stream->tstream, source);

    stream->free		=  antlr3CTSFree;
    return  stream;
}

ANTLR3_API pANTLR3_COMMON_TOKEN_STREAM
antlr3CommonTokenStreamNew(ANTLR3_UINT32 hint)
{
    pANTLR3_COMMON_TOKEN_STREAM stream;

    /* Memory for the interface structure
     */
    stream  = (pANTLR3_COMMON_TOKEN_STREAM) ANTLR3_MALLOC(sizeof(ANTLR3_COMMON_TOKEN_STREAM));

    if	(stream == NULL)
    {
	return	NULL;
    }

    /* Create space for the token stream interface
     */
    stream->tstream	    = antlr3TokenStreamNew();
    stream->tstream->super  =  stream;

    /* Create space for the INT_STREAM interfacce
     */
    stream->tstream->istream		    =  antlr3IntStreamNew();
    stream->tstream->istream->super	    =  (stream->tstream);
    stream->tstream->istream->type	    = ANTLR3_TOKENSTREAM;

    /* Install the token tracking tables
     */
    stream->tokens  = antlr3VectorNew(0);

    /* Defaults
     */
    stream->p	    = -1;

    /* Install the common token stream API
     */
    stream->setTokenTypeChannel	    =  setTokenTypeChannel;
    stream->discardTokenType	    =  discardTokenType;
    stream->discardOffChannelToks   =  discardOffChannel;
    stream->getTokens		    =  getTokens;
    stream->getTokenRange	    =  getTokenRange;
    stream->getTokensSet	    =  getTokensSet;
    stream->getTokensList	    =  getTokensList;
    stream->getTokensType	    =  getTokensType;

    /* Install the token stream API
     */
    stream->tstream->_LT				=  tokLT;
    stream->tstream->get				=  get;
    stream->tstream->getTokenSource		=  getTokenSource;
    stream->tstream->setTokenSource		=  setTokenSource;
    stream->tstream->toString			=  toString;
    stream->tstream->toStringSS			=  toStringSS;
    stream->tstream->toStringTT			=  toStringTT;
	stream->tstream->setDebugListener	=  setDebugListener;

    /* Install INT_STREAM interface
     */
    stream->tstream->istream->_LA	=  _LA;
    stream->tstream->istream->mark	=  mark;
    stream->tstream->istream->release	=  release;
    stream->tstream->istream->size	=  size;
    stream->tstream->istream->index	=  tindex;
    stream->tstream->istream->rewind	=  rewindStream;
    stream->tstream->istream->rewindLast=  rewindLast;
    stream->tstream->istream->seek	=  seek;
    stream->tstream->istream->consume	=  consume;
	stream->tstream->istream->getSourceName = getSourceName;

    return  stream;
}

// Install a debug listener adn switch to debug mode methods
//
static void					
setDebugListener	(pANTLR3_TOKEN_STREAM ts, pANTLR3_DEBUG_EVENT_LISTENER debugger)
{
		// Install the debugger object
	//
	ts->debugger = debugger;

	// Override standard token stream methods with debugging versions
	//
	ts->initialStreamState	= ANTLR3_FALSE;

	ts->_LT				= dbgTokLT;

	ts->istream->consume		= dbgConsume;
	ts->istream->_LA			= dbgLA;
	ts->istream->mark			= dbgMark;
	ts->istream->rewind			= dbgRewindStream;
	ts->istream->rewindLast		= dbgRewindLast;
	ts->istream->seek			= dbgSeek;
}

/** Get the ith token from the current position 1..n where k=1 is the
*  first symbol of lookahead.
*/
static pANTLR3_COMMON_TOKEN 
tokLT  (pANTLR3_TOKEN_STREAM ts, ANTLR3_INT32 k)
{
	ANTLR3_INT32    i;
	ANTLR3_INT32    n;
	pANTLR3_COMMON_TOKEN_STREAM cts;

	cts	    = (pANTLR3_COMMON_TOKEN_STREAM)ts->super;

    if	(k < 0)
	{
		return LB(cts, -k);
	}

	if	(cts->p == -1)
	{
		fillBuffer(cts);
	}
	if	(k == 0)
	{
		return NULL;
	}

	if	((cts->p + k - 1) >= (ANTLR3_INT32)ts->istream->cachedSize)
	{
		pANTLR3_COMMON_TOKEN    teof = &(ts->tokenSource->eofToken);

		teof->setStartIndex (teof, ts->istream->index	    (ts->istream));
		teof->setStopIndex  (teof, ts->istream->index	    (ts->istream));
		return  teof;
	}

	i	= cts->p;
	n	= 1;

	/* Need to find k good tokens, skipping ones that are off channel
	*/
	while   ( n < k)
	{
		/* Skip off-channel tokens */
		i = skipOffTokenChannels(cts, i+1); /* leave p on valid token    */
		n++;
	}
	if	( (ANTLR3_UINT32) i >= ts->istream->cachedSize)
	{
		pANTLR3_COMMON_TOKEN    teof = &(ts->tokenSource->eofToken);

		teof->setStartIndex (teof, ts->istream->index(ts->istream));
		teof->setStopIndex  (teof, ts->istream->index(ts->istream));
		return  teof;
	}

	// Here the token must be in the input vector. Rather then incut
	// function call penalty, we jsut return the pointer directly
	// from the vector
	//
	return  (pANTLR3_COMMON_TOKEN)cts->tokens->elements[i].element;
	//return  (pANTLR3_COMMON_TOKEN)cts->tokens->get(cts->tokens, i);
}

/// Debug only method to flag consumption of initial off-channel
/// tokens in the input stream
///
static void
consumeInitialHiddenTokens(pANTLR3_INT_STREAM is)
{
	ANTLR3_MARKER	first;
	ANTLR3_INT32	i;
	pANTLR3_TOKEN_STREAM	ts;

	ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;
	first	= is->index(is);

	for	(i=0; i<first; i++)
	{
		ts->debugger->consumeHiddenToken(ts->debugger, ts->get(ts, i));
	}

	ts->initialStreamState = ANTLR3_FALSE;

}

/// As per the normal tokLT but sends information to the debugger
///
static pANTLR3_COMMON_TOKEN 
dbgTokLT  (pANTLR3_TOKEN_STREAM ts, ANTLR3_INT32 k)
{
	if	(ts->initialStreamState == ANTLR3_TRUE)
	{
		consumeInitialHiddenTokens(ts->istream);
	}
	return tokLT(ts, k);
}

#ifdef	ANTLR3_WINDOWS
	/* When fully optimized VC7 complains about non reachable code.
	 * Not yet sure if this is an optimizer bug, or a bug in the flow analysis
	 */
#pragma warning( disable : 4702 )
#endif

static pANTLR3_COMMON_TOKEN
LB(pANTLR3_COMMON_TOKEN_STREAM cts, ANTLR3_INT32 k)
{
    ANTLR3_INT32 i;
    ANTLR3_INT32 n;

    if (cts->p == -1)
    {
        fillBuffer(cts);
    }
    if (k == 0)
    {
        return NULL;
    }
    if ((cts->p - k) < 0)
    {
        return NULL;
    }

    i = cts->p;
    n = 1;

    /* Need to find k good tokens, going backwards, skipping ones that are off channel
     */
    while (n <= (ANTLR3_INT32) k)
    {
        /* Skip off-channel tokens
         */

        i = skipOffTokenChannelsReverse(cts, i - 1); /* leave p on valid token    */
        n++;
    }
    if (i < 0)
    {
        return NULL;
    }
	// Here the token must be in the input vector. Rather then incut
	// function call penalty, we jsut return the pointer directly
	// from the vector
	//
	return  (pANTLR3_COMMON_TOKEN)cts->tokens->elements[i].element;
}

static pANTLR3_COMMON_TOKEN 
get (pANTLR3_TOKEN_STREAM ts, ANTLR3_UINT32 i)
{
    pANTLR3_COMMON_TOKEN_STREAM cts;

    cts	    = (pANTLR3_COMMON_TOKEN_STREAM)ts->super;

    return  (pANTLR3_COMMON_TOKEN)(cts->tokens->get(cts->tokens, i));  /* Token index is zero based but vectors are 1 based */
}

static pANTLR3_TOKEN_SOURCE 
getTokenSource	(pANTLR3_TOKEN_STREAM ts)
{
    return  ts->tokenSource;
}

static void
setTokenSource	(   pANTLR3_TOKEN_STREAM ts,
		    pANTLR3_TOKEN_SOURCE tokenSource)
{
    ts->tokenSource	= tokenSource;
}

static pANTLR3_STRING	    
toString    (pANTLR3_TOKEN_STREAM ts)
{
    pANTLR3_COMMON_TOKEN_STREAM cts;

    cts	    = (pANTLR3_COMMON_TOKEN_STREAM)ts->super;

    if	(cts->p == -1)
    {
	fillBuffer(cts);
    }

    return  ts->toStringSS(ts, 0, ts->istream->size(ts->istream));
}

static pANTLR3_STRING
toStringSS(pANTLR3_TOKEN_STREAM ts, ANTLR3_UINT32 start, ANTLR3_UINT32 stop)
{
    pANTLR3_STRING string;
    pANTLR3_TOKEN_SOURCE tsource;
    pANTLR3_COMMON_TOKEN tok;
    ANTLR3_UINT32 i;
    pANTLR3_COMMON_TOKEN_STREAM cts;

    cts = (pANTLR3_COMMON_TOKEN_STREAM) ts->super;

    if (cts->p == -1)
    {
        fillBuffer(cts);
    }
    if (stop >= ts->istream->size(ts->istream))
    {
        stop = ts->istream->size(ts->istream) - 1;
    }

    /* Who is giving us these tokens?
     */
    tsource = ts->getTokenSource(ts);

    if (tsource != NULL && cts->tokens != NULL)
    {
        /* Finally, let's get a string
         */
        string = tsource->strFactory->newRaw(tsource->strFactory);

        for (i = start; i <= stop; i++)
        {
            tok = ts->get(ts, i);
            if (tok != NULL)
            {
                string->appendS(string, tok->getText(tok));
            }
        }

        return string;
    }
    return NULL;

}

static pANTLR3_STRING	    
toStringTT  (pANTLR3_TOKEN_STREAM ts, pANTLR3_COMMON_TOKEN start, pANTLR3_COMMON_TOKEN stop)
{
	if	(start != NULL && stop != NULL)
	{
		return	ts->toStringSS(ts, (ANTLR3_UINT32)start->getTokenIndex(start), (ANTLR3_UINT32)stop->getTokenIndex(stop));
	}
	else
	{
		return	NULL;
	}
}

/** Move the input pointer to the next incoming token.  The stream
 *  must become active with LT(1) available.  consume() simply
 *  moves the input pointer so that LT(1) points at the next
 *  input symbol. Consume at least one token.
 *
 *  Walk past any token not on the channel the parser is listening to.
 */
static void		    
consume	(pANTLR3_INT_STREAM is)
{
	pANTLR3_COMMON_TOKEN_STREAM cts;
	pANTLR3_TOKEN_STREAM	ts;

	ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;
	cts	    = (pANTLR3_COMMON_TOKEN_STREAM) ts->super;

	if	((ANTLR3_UINT32)cts->p < cts->tokens->size(cts->tokens))
	{
		cts->p++;
		cts->p	= skipOffTokenChannels(cts, cts->p);
	}
}


/// As per ordinary consume but notifies the debugger about hidden
/// tokens and so on.
///
static void
dbgConsume	(pANTLR3_INT_STREAM is)
{
	pANTLR3_TOKEN_STREAM	ts;
	ANTLR3_MARKER			a;
	ANTLR3_MARKER			b;
	pANTLR3_COMMON_TOKEN	t;

	ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;

	if	(ts->initialStreamState == ANTLR3_TRUE)
	{
		consumeInitialHiddenTokens(is);
	}
	
	a = is->index(is);		// Where are we right now?
	t = ts->_LT(ts, 1);		// Current token from stream

	consume(is);			// Standard consumer

	b = is->index(is);		// Where are we after consuming 1 on channel token?

	ts->debugger->consumeToken(ts->debugger, t);	// Tell the debugger that we consumed the first token

	if	(b>a+1)
	{
		// The standard consume caused the index to advance by more than 1,
		// which can only happen if it skipped some off-channel tokens.
		// we need to tell the debugger about those tokens.
		//
		ANTLR3_MARKER	i;

		for	(i = a+1; i<b; i++)
		{
			ts->debugger->consumeHiddenToken(ts->debugger, ts->get(ts, (ANTLR3_UINT32)i));
		}

	}
}

/** A simple filter mechanism whereby you can tell this token stream
 *  to force all tokens of type ttype to be on channel.  For example,
 *  when interpreting, we cannot execute actions so we need to tell
 *  the stream to force all WS and NEWLINE to be a different, ignored,
 *  channel.
 */
static void		    
setTokenTypeChannel (pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_UINT32 ttype, ANTLR3_UINT32 channel)
{
    if	(tokenStream->channelOverrides == NULL)
    {
	tokenStream->channelOverrides	= antlr3ListNew(10);
    }

    /* We add one to the channel so we can distinguish NULL as being no entry in the
     * table for a particular token type.
     */
    tokenStream->channelOverrides->put(tokenStream->channelOverrides, ttype, ANTLR3_FUNC_PTR((ANTLR3_UINT32)channel + 1), NULL);
}

static void		    
discardTokenType    (pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_INT32 ttype)
{
    if	(tokenStream->discardSet == NULL)
    {
	tokenStream->discardSet	= antlr3ListNew(31);
    }

    /* We add one to the channel so we can distinguish NULL as being no entry in the
     * table for a particular token type. We could use bitsets for this I suppose too.
     */
    tokenStream->discardSet->put(tokenStream->discardSet, ttype, ANTLR3_FUNC_PTR((ANTLR3_UINT32)ttype + 1), NULL);
}

static void		    
discardOffChannel   (pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_BOOLEAN discard)
{
    tokenStream->discardOffChannel  = discard;
}

static pANTLR3_VECTOR	    
getTokens   (pANTLR3_COMMON_TOKEN_STREAM tokenStream)
{
    if	(tokenStream->p == -1)
    {
	fillBuffer(tokenStream);
    }

    return  tokenStream->tokens;
}

static pANTLR3_LIST	    
getTokenRange	(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_UINT32 start, ANTLR3_UINT32 stop)
{
    return tokenStream->getTokensSet(tokenStream, start, stop, NULL);
}                                                   
/** Given a start and stop index, return a List of all tokens in
 *  the token type BitSet.  Return null if no tokens were found.  This
 *  method looks at both on and off channel tokens.
 */
static pANTLR3_LIST	    
getTokensSet	(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_UINT32 start, ANTLR3_UINT32 stop, pANTLR3_BITSET types)
{
    pANTLR3_LIST	    filteredList;
    ANTLR3_UINT32	    i;
    ANTLR3_UINT32	    n;
    pANTLR3_COMMON_TOKEN    tok;

    if	(tokenStream->p == -1)
    {
	fillBuffer(tokenStream);
    }
    if	(stop > tokenStream->tstream->istream->size(tokenStream->tstream->istream))
    {
	stop = tokenStream->tstream->istream->size(tokenStream->tstream->istream);
    }
    if	(start > stop)
    {
	return NULL;
    }

    /* We have the range set, now we need to iterate through the
     * installed tokens and create a new list with just the ones we want
     * in it. We are just moving pointers about really.
     */
    filteredList    = antlr3ListNew((ANTLR3_UINT32)tokenStream->tstream->istream->size(tokenStream->tstream->istream));

    for	(i = start, n = 0; i<= stop; i++)
    {
	tok = tokenStream->tstream->get(tokenStream->tstream, i);

	if  (	   types == NULL
		|| types->isMember(types, tok->getType(tok) == ANTLR3_TRUE)
	    )
	{
	    filteredList->put(filteredList, n++, (void *)tok, NULL);
	}
    }
    
    /* Did we get any then?
     */
    if	(filteredList->size(filteredList) == 0)
    {
	filteredList->free(filteredList);
	filteredList	= NULL;
    }

    return  filteredList;
}

static pANTLR3_LIST	    
getTokensList	(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_UINT32 start, ANTLR3_UINT32 stop, pANTLR3_LIST list)
{
    pANTLR3_BITSET  bitSet;
    pANTLR3_LIST    newlist;

    bitSet  = antlr3BitsetList(list->table);

    newlist    = tokenStream->getTokensSet(tokenStream, start, stop, bitSet);

    bitSet->free(bitSet);

    return  newlist;

}

static pANTLR3_LIST	    
getTokensType	(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_UINT32 start, ANTLR3_UINT32 stop, ANTLR3_UINT32 type)
{
    pANTLR3_BITSET  bitSet;
    pANTLR3_LIST    newlist;

    bitSet  = antlr3BitsetOf(type, -1);
    newlist = tokenStream->getTokensSet(tokenStream, start, stop, bitSet);

    bitSet->free(bitSet);

    return  newlist;
}

static ANTLR3_UINT32	    
_LA  (pANTLR3_INT_STREAM is, ANTLR3_INT32 i)
{
	pANTLR3_TOKEN_STREAM    ts;
	pANTLR3_COMMON_TOKEN    tok;

	ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;

	tok	    =  ts->_LT(ts, i);

	if	(tok != NULL)
	{
		return	tok->getType(tok);
	}
	else
	{
		return	ANTLR3_TOKEN_INVALID;
	}
}

/// As per _LA() but for debug mode.
///
static ANTLR3_UINT32	    
dbgLA  (pANTLR3_INT_STREAM is, ANTLR3_INT32 i)
{
    pANTLR3_TOKEN_STREAM    ts;
   
    ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;

	if	(ts->initialStreamState == ANTLR3_TRUE)
	{
		consumeInitialHiddenTokens(is);
	}
	ts->debugger->LT(ts->debugger, i, tokLT(ts, i));
	return	_LA(is, i);
}

static ANTLR3_MARKER
mark	(pANTLR3_INT_STREAM is)
{
    is->lastMarker = is->index(is);
    return  is->lastMarker;
}

/// As per mark() but with a call to tell the debugger we are doing this
///
static ANTLR3_MARKER
dbgMark	(pANTLR3_INT_STREAM is)
{
    pANTLR3_TOKEN_STREAM    ts;
   
    ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;
	
	is->lastMarker = is->index(is);
	ts->debugger->mark(ts->debugger, is->lastMarker);

    return  is->lastMarker;
}

static void		    
release	(pANTLR3_INT_STREAM is, ANTLR3_MARKER mark)
{
    return;
}

static ANTLR3_UINT32	    
size	(pANTLR3_INT_STREAM is)
{
    pANTLR3_COMMON_TOKEN_STREAM cts;
    pANTLR3_TOKEN_STREAM	ts;

    if (is->cachedSize > 0)
    {
	return  is->cachedSize;
    }
    ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;
    cts	    = (pANTLR3_COMMON_TOKEN_STREAM) ts->super;

    is->cachedSize =  cts->tokens->count;
    return  is->cachedSize;
}

static ANTLR3_MARKER   
tindex	(pANTLR3_INT_STREAM is)
{
    pANTLR3_COMMON_TOKEN_STREAM cts;
    pANTLR3_TOKEN_STREAM	ts;

    ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;
    cts	    = (pANTLR3_COMMON_TOKEN_STREAM) ts->super;

    return  cts->p;
}

static void		    
dbgRewindLast	(pANTLR3_INT_STREAM is)
{
	pANTLR3_TOKEN_STREAM	ts;

    ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;

	ts->debugger->rewindLast(ts->debugger);

    is->rewind(is, is->lastMarker);
}
static void		    
rewindLast	(pANTLR3_INT_STREAM is)
{
    is->rewind(is, is->lastMarker);
}
static void		    
rewindStream	(pANTLR3_INT_STREAM is, ANTLR3_MARKER marker)
{
    is->seek(is, (ANTLR3_UINT32)(marker));
}
static void		    
dbgRewindStream	(pANTLR3_INT_STREAM is, ANTLR3_MARKER marker)
{
    pANTLR3_TOKEN_STREAM	ts;

    ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;

	ts->debugger->rewind(ts->debugger, marker);

    is->seek(is, (ANTLR3_UINT32)(marker));
}

static void		    
seek	(pANTLR3_INT_STREAM is, ANTLR3_MARKER index)
{
    pANTLR3_COMMON_TOKEN_STREAM cts;
    pANTLR3_TOKEN_STREAM	ts;

    ts	    = (pANTLR3_TOKEN_STREAM)	    is->super;
    cts	    = (pANTLR3_COMMON_TOKEN_STREAM) ts->super;

    cts->p  = (ANTLR3_UINT32)index;
}
static void		    
dbgSeek	(pANTLR3_INT_STREAM is, ANTLR3_MARKER index)
{
	// TODO: Implement seek in debugger when Ter adds it to Java
	//
	seek(is, index);
}
ANTLR3_API void
fillBufferExt(pANTLR3_COMMON_TOKEN_STREAM tokenStream)
{
    fillBuffer(tokenStream);
}
static void
fillBuffer(pANTLR3_COMMON_TOKEN_STREAM tokenStream) {
    ANTLR3_UINT32 index;
    pANTLR3_COMMON_TOKEN tok;
    ANTLR3_BOOLEAN discard;
    void * channelI;

    /* Start at index 0 of course
     */
    index = 0;

    /* Pick out the next token from the token source
     * Remember we just get a pointer (reference if you like) here
     * and so if we store it anywhere, we don't set any pointers to auto free it.
     */
    tok = tokenStream->tstream->tokenSource->nextToken(tokenStream->tstream->tokenSource);

    while (tok != NULL && tok->type != ANTLR3_TOKEN_EOF)
    {
        discard = ANTLR3_FALSE; /* Assume we are not discarding	*/

        /* I employ a bit of a trick, or perhaps hack here. Rather than
         * store a pointer to a structure in the override map and discard set
         * we store the value + 1 cast to a void *. Hence on systems where NULL = (void *)0
         * we can distinguish "not being there" from "being channel or type 0"
         */

        if (tokenStream->discardSet != NULL
            && tokenStream->discardSet->get(tokenStream->discardSet, tok->getType(tok)) != NULL)
        {
            discard = ANTLR3_TRUE;
        }
        else if (   tokenStream->discardOffChannel == ANTLR3_TRUE
                 && tok->getChannel(tok) != tokenStream->channel
                 )
        {
            discard = ANTLR3_TRUE;
        }
        else if (tokenStream->channelOverrides != NULL)
        {
            /* See if this type is in the override map
             */
            channelI = tokenStream->channelOverrides->get(tokenStream->channelOverrides, tok->getType(tok) + 1);

            if (channelI != NULL)
            {
                /* Override found
                 */
                tok->setChannel(tok, ANTLR3_UINT32_CAST(channelI) - 1);
            }
        }

        /* If not discarding it, add it to the list at the current index
         */
        if (discard == ANTLR3_FALSE)
        {
            /* Add it, indicating that we will delete it and the table should not
             */
            tok->setTokenIndex(tok, index);
            tokenStream->p++;
            tokenStream->tokens->add(tokenStream->tokens, (void *) tok, NULL);
            index++;
        }

        tok = tokenStream->tstream->tokenSource->nextToken(tokenStream->tstream->tokenSource);
    }

    /* Cache the size so we don't keep doing indirect method calls. We do this as
     * early as possible so that anything after this may utilize the cached value.
     */
    tokenStream->tstream->istream->cachedSize = tokenStream->tokens->count;

    /* Set the consume pointer to the first token that is on our channel
     */
    tokenStream->p = 0;
    tokenStream->p = skipOffTokenChannels(tokenStream, tokenStream->p);

}

/// Given a starting index, return the index of the first on-channel
///  token.
///
static ANTLR3_UINT32
skipOffTokenChannels(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_INT32 i) {
    ANTLR3_INT32 n;
    pANTLR3_COMMON_TOKEN tok;

    n = tokenStream->tstream->istream->cachedSize;

    while (i < n)
    {
        tok =  (pANTLR3_COMMON_TOKEN)tokenStream->tokens->elements[i].element;

        if (tok->channel!= tokenStream->channel)
        {
            i++;
        }
        else
        {
            return i;
        }
    }
    return i;
}

static ANTLR3_UINT32
skipOffTokenChannelsReverse(pANTLR3_COMMON_TOKEN_STREAM tokenStream, ANTLR3_INT32 x)
{
    pANTLR3_COMMON_TOKEN tok;

    while (x >= 0)
    {
        tok =  (pANTLR3_COMMON_TOKEN)tokenStream->tokens->elements[x].element;
        
        if ((tok->channel != tokenStream->channel))
        {
            x--;
        }
        else
        {
            return x;
        }
    }
    return x;
}

/// Return a string that represents the name assoicated with the input source
///
/// /param[in] is The ANTLR3_INT_STREAM interface that is representing this token stream.
///
/// /returns 
/// /implements ANTLR3_INT_STREAM_struct::getSourceName()
///
static pANTLR3_STRING		
getSourceName				(pANTLR3_INT_STREAM is)
{
	// Slightly convoluted as we must trace back to the lexer's input source
	// via the token source. The streamName that is here is not initialized
	// because this is a token stream, not a file or string stream, which are the
	// only things that have a context for a source name.
	//
	return ((pANTLR3_TOKEN_STREAM)(is->super))->tokenSource->fileName;
}
