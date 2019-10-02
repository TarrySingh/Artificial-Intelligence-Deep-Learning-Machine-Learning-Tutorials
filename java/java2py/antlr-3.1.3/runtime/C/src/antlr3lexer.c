/** \file
 *
 * Base implementation of an antlr 3 lexer.
 *
 * An ANTLR3 lexer implements a base recongizer, a token source and
 * a lexer interface. It constructs a base recognizer with default
 * functions, then overrides any of these that are parser specific (usual
 * default implementation of base recognizer.
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

#include    <antlr3lexer.h>

static void					mTokens						(pANTLR3_LEXER lexer);
static void					setCharStream				(pANTLR3_LEXER lexer,  pANTLR3_INPUT_STREAM input);
static void					pushCharStream				(pANTLR3_LEXER lexer,  pANTLR3_INPUT_STREAM input);
static void					popCharStream				(pANTLR3_LEXER lexer);

static void					emitNew						(pANTLR3_LEXER lexer,  pANTLR3_COMMON_TOKEN token);
static pANTLR3_COMMON_TOKEN emit						(pANTLR3_LEXER lexer);
static ANTLR3_BOOLEAN	    matchs						(pANTLR3_LEXER lexer, ANTLR3_UCHAR * string);
static ANTLR3_BOOLEAN	    matchc						(pANTLR3_LEXER lexer, ANTLR3_UCHAR c);
static ANTLR3_BOOLEAN	    matchRange					(pANTLR3_LEXER lexer, ANTLR3_UCHAR low, ANTLR3_UCHAR high);
static void					matchAny					(pANTLR3_LEXER lexer);
static void					recover						(pANTLR3_LEXER lexer);
static ANTLR3_UINT32	    getLine						(pANTLR3_LEXER lexer);
static ANTLR3_MARKER	    getCharIndex				(pANTLR3_LEXER lexer);
static ANTLR3_UINT32	    getCharPositionInLine		(pANTLR3_LEXER lexer);
static pANTLR3_STRING	    getText						(pANTLR3_LEXER lexer);
static pANTLR3_COMMON_TOKEN nextToken					(pANTLR3_TOKEN_SOURCE toksource);

static void					displayRecognitionError	    (pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 * tokenNames);
static void					reportError					(pANTLR3_BASE_RECOGNIZER rec);
static void *				getCurrentInputSymbol		(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM istream);
static void *				getMissingSymbol			(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM	istream, pANTLR3_EXCEPTION	e,
															ANTLR3_UINT32 expectedTokenType, pANTLR3_BITSET_LIST follow);

static void					reset						(pANTLR3_BASE_RECOGNIZER rec);

static void					freeLexer					(pANTLR3_LEXER lexer);


ANTLR3_API pANTLR3_LEXER
antlr3LexerNew(ANTLR3_UINT32 sizeHint, pANTLR3_RECOGNIZER_SHARED_STATE state)
{
    pANTLR3_LEXER   lexer;
    pANTLR3_COMMON_TOKEN	specialT;

	/* Allocate memory
	*/
	lexer   = (pANTLR3_LEXER) ANTLR3_MALLOC(sizeof(ANTLR3_LEXER));

	if	(lexer == NULL)
	{
		return	NULL;
	}

	/* Now we need to create the base recognizer
	*/
	lexer->rec	    =  antlr3BaseRecognizerNew(ANTLR3_TYPE_LEXER, sizeHint, state);

	if	(lexer->rec == NULL)
	{
		lexer->free(lexer);
		return	NULL;
	}
	lexer->rec->super  =  lexer;

	lexer->rec->displayRecognitionError	    = displayRecognitionError;
	lexer->rec->reportError					= reportError;
	lexer->rec->reset						= reset;
	lexer->rec->getCurrentInputSymbol		= getCurrentInputSymbol;
	lexer->rec->getMissingSymbol			= getMissingSymbol;

	/* Now install the token source interface
	*/
	if	(lexer->rec->state->tokSource == NULL) 
	{
		lexer->rec->state->tokSource	= (pANTLR3_TOKEN_SOURCE)ANTLR3_MALLOC(sizeof(ANTLR3_TOKEN_SOURCE));

		if	(lexer->rec->state->tokSource == NULL) 
		{
			lexer->rec->free(lexer->rec);
			lexer->free(lexer);

			return	NULL;
		}
		lexer->rec->state->tokSource->super    =  lexer;

		/* Install the default nextToken() method, which may be overridden
		 * by generated code, or by anything else in fact.
		 */
		lexer->rec->state->tokSource->nextToken	    =  nextToken;
		lexer->rec->state->tokSource->strFactory    = NULL;

		lexer->rec->state->tokFactory				= NULL;
	}

    /* Install the lexer API
     */
    lexer->setCharStream			=  setCharStream;
    lexer->mTokens					= (void (*)(void *))(mTokens);
    lexer->setCharStream			=  setCharStream;
    lexer->pushCharStream			=  pushCharStream;
    lexer->popCharStream			=  popCharStream;
    lexer->emit						=  emit;
    lexer->emitNew					=  emitNew;
    lexer->matchs					=  matchs;
    lexer->matchc					=  matchc;
    lexer->matchRange				=  matchRange;
    lexer->matchAny					=  matchAny;
    lexer->recover					=  recover;
    lexer->getLine					=  getLine;
    lexer->getCharIndex				=  getCharIndex;
    lexer->getCharPositionInLine    =  getCharPositionInLine;
    lexer->getText					=  getText;
    lexer->free						=  freeLexer;
    
    /* Initialise the eof token
     */
    specialT				= &(lexer->rec->state->tokSource->eofToken);
    antlr3SetTokenAPI	  (specialT);
    specialT->setType	  (specialT, ANTLR3_TOKEN_EOF);
    specialT->factoryMade	= ANTLR3_TRUE;					// Prevent things trying to free() it
    specialT->strFactory    = NULL;

	// Initialize the skip token.
	//
    specialT				= &(lexer->rec->state->tokSource->skipToken);
    antlr3SetTokenAPI	  (specialT);
    specialT->setType	  (specialT, ANTLR3_TOKEN_INVALID);
    specialT->factoryMade	= ANTLR3_TRUE;					// Prevent things trying to free() it
    specialT->strFactory    = NULL;
    return  lexer;
}

static void
reset	(pANTLR3_BASE_RECOGNIZER rec)
{
    pANTLR3_LEXER   lexer;

    lexer   = rec->super;

    lexer->rec->state->token			= NULL;
    lexer->rec->state->type				= ANTLR3_TOKEN_INVALID;
    lexer->rec->state->channel			= ANTLR3_TOKEN_DEFAULT_CHANNEL;
    lexer->rec->state->tokenStartCharIndex		= -1;
    lexer->rec->state->tokenStartCharPositionInLine = -1;
    lexer->rec->state->tokenStartLine		= -1;

    lexer->rec->state->text	    = NULL;

    if (lexer->input != NULL)
    {
	lexer->input->istream->seek(lexer->input->istream, 0);
    }
}

///
/// \brief
/// Returns the next available token from the current input stream.
/// 
/// \param toksource
/// Points to the implementation of a token source. The lexer is 
/// addressed by the super structure pointer.
/// 
/// \returns
/// The next token in the current input stream or the EOF token
/// if there are no more tokens.
/// 
/// \remarks
/// Write remarks for nextToken here.
/// 
/// \see nextToken
///
ANTLR3_INLINE static pANTLR3_COMMON_TOKEN
nextTokenStr	    (pANTLR3_TOKEN_SOURCE toksource)
{
	pANTLR3_LEXER   lexer;

	lexer   = (pANTLR3_LEXER)(toksource->super);

	/// Loop until we get a non skipped token or EOF
	///
	for	(;;)
	{
		// Get rid of any previous token (token factory takes care of
		// any de-allocation when this token is finally used up.
		//
		lexer->rec->state->token		    = NULL;
		lexer->rec->state->error		    = ANTLR3_FALSE;	    // Start out without an exception
		lexer->rec->state->failed		    = ANTLR3_FALSE;



		// Now call the matching rules and see if we can generate a new token
		//
		for	(;;)
		{
            // Record the start of the token in our input stream.
            //
            lexer->rec->state->channel						= ANTLR3_TOKEN_DEFAULT_CHANNEL;
            lexer->rec->state->tokenStartCharIndex			= lexer->input->istream->index(lexer->input->istream);
            lexer->rec->state->tokenStartCharPositionInLine	= lexer->input->getCharPositionInLine(lexer->input);
        	lexer->rec->state->tokenStartLine				= lexer->input->getLine(lexer->input);
            lexer->rec->state->text							= NULL;

			if  (lexer->input->istream->_LA(lexer->input->istream, 1) == ANTLR3_CHARSTREAM_EOF)
			{
				// Reached the end of the current stream, nothing more to do if this is
				// the last in the stack.
				//
				pANTLR3_COMMON_TOKEN    teof = &(toksource->eofToken);

				teof->setStartIndex (teof, lexer->getCharIndex(lexer));
				teof->setStopIndex  (teof, lexer->getCharIndex(lexer));
				teof->setLine	(teof, lexer->getLine(lexer));
				teof->factoryMade = ANTLR3_TRUE;	// This isn't really manufactured but it stops things from trying to free it
				return  teof;
			}

			lexer->rec->state->token		= NULL;
			lexer->rec->state->error		= ANTLR3_FALSE;	    // Start out without an exception
			lexer->rec->state->failed		= ANTLR3_FALSE;

			// Call the generated lexer, see if it can get a new token together.
			//
			lexer->mTokens(lexer->ctx);

			if  (lexer->rec->state->error  == ANTLR3_TRUE)
			{
				// Recognition exception, report it and try to recover.
				//
				lexer->rec->state->failed	    = ANTLR3_TRUE;
				lexer->rec->reportError(lexer->rec);
				lexer->recover(lexer); 
			}
			else
			{
				if (lexer->rec->state->token == NULL)
				{
					// Emit the real token, which adds it in to the token stream basically
					//
					emit(lexer);
				}
				else if	(lexer->rec->state->token ==  &(toksource->skipToken))
				{
					// A real token could have been generated, but "Computer say's naaaaah" and it
					// it is just something we need to skip altogether.
					//
					continue;
				}
				
				// Good token, not skipped, not EOF token
				//
				return  lexer->rec->state->token;
			}
		}
	}
}

/**
 * \brief
 * Default implementation of the nextToken() call for a lexer.
 * 
 * \param toksource
 * Points to the implementation of a token source. The lexer is 
 * addressed by the super structure pointer.
 * 
 * \returns
 * The next token in the current input stream or the EOF token
 * if there are no more tokens in any input stream in the stack.
 * 
 * Write detailed description for nextToken here.
 * 
 * \remarks
 * Write remarks for nextToken here.
 * 
 * \see nextTokenStr
 */
static pANTLR3_COMMON_TOKEN
nextToken	    (pANTLR3_TOKEN_SOURCE toksource)
{
	pANTLR3_COMMON_TOKEN tok;

	// Find the next token in the current stream
	//
	tok = nextTokenStr(toksource);

	// If we got to the EOF token then switch to the previous
	// input stream if there were any and just return the
	// EOF if there are none. We must check the next token
	// in any outstanding input stream we pop into the active
	// role to see if it was sitting at EOF after PUSHing the
	// stream we just consumed, otherwise we will return EOF
	// on the reinstalled input stream, when in actual fact
	// there might be more input streams to POP before the
	// real EOF of the whole logical inptu stream. Hence we
	// use a while loop here until we find somethign in the stream
	// that isn't EOF or we reach the actual end of the last input
	// stream on the stack.
	//
	while	(tok->type == ANTLR3_TOKEN_EOF)
	{
		pANTLR3_LEXER   lexer;

		lexer   = (pANTLR3_LEXER)(toksource->super);

		if  (lexer->rec->state->streams != NULL && lexer->rec->state->streams->size(lexer->rec->state->streams) > 0)
		{
			// We have another input stream in the stack so we
			// need to revert to it, then resume the loop to check
			// it wasn't sitting at EOF itself.
			//
			lexer->popCharStream(lexer);
			tok = nextTokenStr(toksource);
		}
		else
		{
			// There were no more streams on the input stack
			// so this EOF is the 'real' logical EOF for
			// the input stream. So we just exit the loop and 
			// return the EOF we have found.
			//
			break;
		}
		
	}

	// return whatever token we have, which may be EOF
	//
	return  tok;
}

ANTLR3_API pANTLR3_LEXER
antlr3LexerNewStream(ANTLR3_UINT32 sizeHint, pANTLR3_INPUT_STREAM input, pANTLR3_RECOGNIZER_SHARED_STATE state)
{
    pANTLR3_LEXER   lexer;

    // Create a basic lexer first
    //
    lexer   = antlr3LexerNew(sizeHint, state);

    if	(lexer != NULL) 
    {
		// Install the input stream and reset the lexer
		//
		setCharStream(lexer, input);
    }

    return  lexer;
}

static void mTokens	    (pANTLR3_LEXER lexer)
{
    if	(lexer)	    // Fool compiler, avoid pragmas
    {
		ANTLR3_FPRINTF(stderr, "lexer->mTokens(): Error: No lexer rules were added to the lexer yet!\n");
    }
}

static void			
reportError		    (pANTLR3_BASE_RECOGNIZER rec)
{
    rec->displayRecognitionError(rec, rec->state->tokenNames);
}

#ifdef	ANTLR3_WINDOWS
#pragma warning( disable : 4100 )
#endif

/** Default lexer error handler (works for 8 bit streams only!!!)
 */
static void			
displayRecognitionError	    (pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_UINT8 * tokenNames)
{
    pANTLR3_LEXER			lexer;
	pANTLR3_EXCEPTION	    ex;
	pANTLR3_STRING			ftext;

    lexer   = (pANTLR3_LEXER)(recognizer->super);
	ex		= lexer->rec->state->exception;

	// See if there is a 'filename' we can use
    //
    if	(ex->name == NULL)
    {
		ANTLR3_FPRINTF(stderr, "-unknown source-(");
    }
    else
    {
		ftext = ex->streamName->to8(ex->streamName);
		ANTLR3_FPRINTF(stderr, "%s(", ftext->chars);
    }

    ANTLR3_FPRINTF(stderr, "%d) ", recognizer->state->exception->line);
    ANTLR3_FPRINTF(stderr, ": lexer error %d :\n\t%s at offset %d, ", 
						ex->type,
						(pANTLR3_UINT8)	   (ex->message),
					    ex->charPositionInLine+1
		    );
	{
		ANTLR3_INT32	width;

		width	= ANTLR3_UINT32_CAST(( (pANTLR3_UINT8)(lexer->input->data) + (lexer->input->size(lexer->input) )) - (pANTLR3_UINT8)(ex->index));

		if	(width >= 1)
		{			
			if	(isprint(ex->c))
			{
				ANTLR3_FPRINTF(stderr, "near '%c' :\n", ex->c);
			}
			else
			{
				ANTLR3_FPRINTF(stderr, "near char(%#02X) :\n", (ANTLR3_UINT8)(ex->c));
			}
			ANTLR3_FPRINTF(stderr, "\t%.*s\n", width > 20 ? 20 : width ,((pANTLR3_UINT8)ex->index));
		}
		else
		{
			ANTLR3_FPRINTF(stderr, "(end of input).\n\t This indicates a poorly specified lexer RULE\n\t or unterminated input element such as: \"STRING[\"]\n");
			ANTLR3_FPRINTF(stderr, "\t The lexer was matching from line %d, offset %d, which\n\t ", 
								(ANTLR3_UINT32)(lexer->rec->state->tokenStartLine),
								(ANTLR3_UINT32)(lexer->rec->state->tokenStartCharPositionInLine)
								);
			width = ANTLR3_UINT32_CAST(((pANTLR3_UINT8)(lexer->input->data)+(lexer->input->size(lexer->input))) - (pANTLR3_UINT8)(lexer->rec->state->tokenStartCharIndex));

			if	(width >= 1)
			{
				ANTLR3_FPRINTF(stderr, "looks like this:\n\t\t%.*s\n", width > 20 ? 20 : width ,(pANTLR3_UINT8)(lexer->rec->state->tokenStartCharIndex));
			}
			else
			{
				ANTLR3_FPRINTF(stderr, "is also the end of the line, so you must check your lexer rules\n");
			}
		}
	}
}

static void setCharStream   (pANTLR3_LEXER lexer,  pANTLR3_INPUT_STREAM input)
{
    /* Install the input interface
     */
    lexer->input	= input;

    /* We may need a token factory for the lexer; we don't destroy any existing factory
     * until the lexer is destroyed, as people may still be using the tokens it produced.
     * TODO: Later I will provide a dup() method for a token so that it can extract itself
     * out of the factory. 
     */
    if	(lexer->rec->state->tokFactory == NULL)
    {
	lexer->rec->state->tokFactory	= antlr3TokenFactoryNew(input);
    }
    else
    {
	/* When the input stream is being changed on the fly, rather than
	 * at the start of a new lexer, then we must tell the tokenFactory
	 * which input stream to adorn the tokens with so that when they
	 * are asked to provide their original input strings they can
	 * do so from the correct text stream.
	 */
	lexer->rec->state->tokFactory->setInputStream(lexer->rec->state->tokFactory, input);
    }

    /* Propagate the string factory so that we preserve the encoding form from
     * the input stream.
     */
    if	(lexer->rec->state->tokSource->strFactory == NULL)
    {
        lexer->rec->state->tokSource->strFactory	= input->strFactory;

        // Set the newly acquired string factory up for our pre-made tokens
        // for EOF.
        //
        if (lexer->rec->state->tokSource->eofToken.strFactory == NULL)
        {
            lexer->rec->state->tokSource->eofToken.strFactory = input->strFactory;
        }
    }

    /* This is a lexer, install the appropriate exception creator
     */
    lexer->rec->exConstruct = antlr3RecognitionExceptionNew;

    /* Set the current token to nothing
     */
    lexer->rec->state->token		= NULL;
    lexer->rec->state->text			= NULL;
    lexer->rec->state->tokenStartCharIndex	= -1;

    /* Copy the name of the char stream to the token source
     */
    lexer->rec->state->tokSource->fileName = input->fileName;
}

/*!
 * \brief
 * Change to a new input stream, remembering the old one.
 * 
 * \param lexer
 * Pointer to the lexer instance to switch input streams for.
 * 
 * \param input
 * New input stream to install as the current one.
 * 
 * Switches the current character input stream to 
 * a new one, saving the old one, which we will revert to at the end of this 
 * new one.
 */
static void
pushCharStream  (pANTLR3_LEXER lexer,  pANTLR3_INPUT_STREAM input)
{
	// Do we need a new input stream stack?
	//
	if	(lexer->rec->state->streams == NULL)
	{
		// This is the first call to stack a new
		// stream and so we must create the stack first.
		//
		lexer->rec->state->streams = antlr3StackNew(0);

		if  (lexer->rec->state->streams == NULL)
		{
			// Could not do this, we just fail to push it.
			// TODO: Consider if this is what we want to do, but then
			//       any programmer can override this method to do something else.
			return;
		}
	}

	// We have a stack, so we can save the current input stream
	// into it.
	//
	lexer->input->istream->mark(lexer->input->istream);
	lexer->rec->state->streams->push(lexer->rec->state->streams, lexer->input, NULL);

	// And now we can install this new one
	//
	lexer->setCharStream(lexer, input);
}

/*!
 * \brief
 * Stops using the current input stream and reverts to any prior
 * input stream on the stack.
 * 
 * \param lexer
 * Description of parameter lexer.
 * 
 * Pointer to a function that abandons the current input stream, whether it
 * is empty or not and reverts to the previous stacked input stream.
 *
 * \remark
 * The function fails silently if there are no prior input streams.
 */
static void
popCharStream   (pANTLR3_LEXER lexer)
{
    pANTLR3_INPUT_STREAM input;

    // If we do not have a stream stack or we are already at the
    // stack bottom, then do nothing.
    //
    if	(lexer->rec->state->streams != NULL && lexer->rec->state->streams->size(lexer->rec->state->streams) > 0)
    {
	// We just leave the current stream to its fate, we do not close
	// it or anything as we do not know what the programmer intended
	// for it. This method can always be overridden of course.
	// So just find out what was currently saved on the stack and use
	// that now, then pop it from the stack.
	//
	input	= (pANTLR3_INPUT_STREAM)(lexer->rec->state->streams->top);
	lexer->rec->state->streams->pop(lexer->rec->state->streams);

	// Now install the stream as the current one.
	//
	lexer->setCharStream(lexer, input);
	lexer->input->istream->rewindLast(lexer->input->istream);
    }
    return;
}

static void emitNew	    (pANTLR3_LEXER lexer,  pANTLR3_COMMON_TOKEN token)
{
    lexer->rec->state->token    = token;	/* Voila!   */
}

static pANTLR3_COMMON_TOKEN
emit	    (pANTLR3_LEXER lexer)
{
    pANTLR3_COMMON_TOKEN	token;

    /* We could check pointers to token factories and so on, but
     * we are in code that we want to run as fast as possible
     * so we are not checking any errors. So make sure you have installed an input stream before
     * trying to emit a new token.
     */
    token   = lexer->rec->state->tokFactory->newToken(lexer->rec->state->tokFactory);

    /* Install the supplied information, and some other bits we already know
     * get added automatically, such as the input stream it is associated with
     * (though it can all be overridden of course)
     */
    token->type		    = lexer->rec->state->type;
    token->channel	    = lexer->rec->state->channel;
    token->start	    = lexer->rec->state->tokenStartCharIndex;
    token->stop		    = lexer->getCharIndex(lexer) - 1;
    token->line		    = lexer->rec->state->tokenStartLine;
    token->charPosition	= lexer->rec->state->tokenStartCharPositionInLine;

	if	(lexer->rec->state->text != NULL)
	{
		token->textState		= ANTLR3_TEXT_STRING;
		token->tokText.text	    = lexer->rec->state->text;
	}
	else
	{
		token->textState	= ANTLR3_TEXT_NONE;
	}
    token->lineStart	= lexer->input->currentLine;
	token->user1		= lexer->rec->state->user1;
	token->user2		= lexer->rec->state->user2;
	token->user3		= lexer->rec->state->user3;
	token->custom		= lexer->rec->state->custom;

    lexer->rec->state->token	    = token;

    return  token;
}

/**
 * Free the resources allocated by a lexer
 */
static void 
freeLexer    (pANTLR3_LEXER lexer)
{
	// This may have ben a delegate or delegator lexer, in which case the
	// state may already have been freed (and set to NULL therefore)
	// so we ignore the state if we don't have it.
	//
	if	(lexer->rec->state != NULL)
	{
		if	(lexer->rec->state->streams != NULL)
		{
			lexer->rec->state->streams->free(lexer->rec->state->streams);
		}
		if	(lexer->rec->state->tokFactory != NULL)
		{
			lexer->rec->state->tokFactory->close(lexer->rec->state->tokFactory);
			lexer->rec->state->tokFactory = NULL;
		}
		if	(lexer->rec->state->tokSource != NULL)
		{
			ANTLR3_FREE(lexer->rec->state->tokSource);
			lexer->rec->state->tokSource = NULL;
		}
	}
	if	(lexer->rec != NULL)
	{
		lexer->rec->free(lexer->rec);
		lexer->rec = NULL;
	}
	ANTLR3_FREE(lexer);
}

/** Implementation of matchs for the lexer, overrides any
 *  base implementation in the base recognizer. 
 *
 *  \remark
 *  Note that the generated code lays down arrays of ints for constant
 *  strings so that they are int UTF32 form!
 */
static ANTLR3_BOOLEAN
matchs(pANTLR3_LEXER lexer, ANTLR3_UCHAR * string)
{
	while   (*string != ANTLR3_STRING_TERMINATOR)
	{
		if  (lexer->input->istream->_LA(lexer->input->istream, 1) != (*string))
		{
			if	(lexer->rec->state->backtracking > 0)
			{
				lexer->rec->state->failed = ANTLR3_TRUE;
				return ANTLR3_FALSE;
			}

			lexer->rec->exConstruct(lexer->rec);
			lexer->rec->state->failed	 = ANTLR3_TRUE;

			/* TODO: Implement exception creation more fully perhaps
			 */
			lexer->recover(lexer);
			return  ANTLR3_FALSE;
		}

		/* Matched correctly, do consume it
		 */
		lexer->input->istream->consume(lexer->input->istream);
		string++;

		/* Reset any failed indicator
		 */
		lexer->rec->state->failed = ANTLR3_FALSE;
	}


	return  ANTLR3_TRUE;
}

/** Implementation of matchc for the lexer, overrides any
 *  base implementation in the base recognizer. 
 *
 *  \remark
 *  Note that the generated code lays down arrays of ints for constant
 *  strings so that they are int UTF32 form!
 */
static ANTLR3_BOOLEAN
matchc(pANTLR3_LEXER lexer, ANTLR3_UCHAR c)
{
	if	(lexer->input->istream->_LA(lexer->input->istream, 1) == c)
	{
		/* Matched correctly, do consume it
		 */
		lexer->input->istream->consume(lexer->input->istream);

		/* Reset any failed indicator
		 */
		lexer->rec->state->failed = ANTLR3_FALSE;

		return	ANTLR3_TRUE;
	}

	/* Failed to match, exception and recovery time.
	 */
	if	(lexer->rec->state->backtracking > 0)
	{
		lexer->rec->state->failed  = ANTLR3_TRUE;
		return	ANTLR3_FALSE;
	}

	lexer->rec->exConstruct(lexer->rec);

	/* TODO: Implement exception creation more fully perhaps
	 */
	lexer->recover(lexer);

	return  ANTLR3_FALSE;
}

/** Implementation of match range for the lexer, overrides any
 *  base implementation in the base recognizer. 
 *
 *  \remark
 *  Note that the generated code lays down arrays of ints for constant
 *  strings so that they are int UTF32 form!
 */
static ANTLR3_BOOLEAN
matchRange(pANTLR3_LEXER lexer, ANTLR3_UCHAR low, ANTLR3_UCHAR high)
{
    ANTLR3_UCHAR    c;

    /* What is in the stream at the moment?
     */
    c	= lexer->input->istream->_LA(lexer->input->istream, 1);
    if	( c >= low && c <= high)
    {
	/* Matched correctly, consume it
	 */
	lexer->input->istream->consume(lexer->input->istream);

	/* Reset any failed indicator
	 */
	lexer->rec->state->failed = ANTLR3_FALSE;

	return	ANTLR3_TRUE;
    }
    
    /* Failed to match, execption and recovery time.
     */

    if	(lexer->rec->state->backtracking > 0)
    {
	lexer->rec->state->failed  = ANTLR3_TRUE;
	return	ANTLR3_FALSE;
    }

    lexer->rec->exConstruct(lexer->rec);

    /* TODO: Implement exception creation more fully
     */
    lexer->recover(lexer);

    return  ANTLR3_FALSE;
}

static void
matchAny	    (pANTLR3_LEXER lexer)
{
    lexer->input->istream->consume(lexer->input->istream);
}

static void
recover	    (pANTLR3_LEXER lexer)
{
    lexer->input->istream->consume(lexer->input->istream);
}

static ANTLR3_UINT32
getLine	    (pANTLR3_LEXER lexer)
{
    return  lexer->input->getLine(lexer->input);
}

static ANTLR3_UINT32
getCharPositionInLine	(pANTLR3_LEXER lexer)
{
    return  lexer->input->getCharPositionInLine(lexer->input);
}

static ANTLR3_MARKER	getCharIndex	    (pANTLR3_LEXER lexer)
{
    return lexer->input->istream->index(lexer->input->istream);
}

static pANTLR3_STRING
getText	    (pANTLR3_LEXER lexer)
{
	if (lexer->rec->state->text)
	{
		return	lexer->rec->state->text;

	}
	return  lexer->input->substr(
									lexer->input, 
									lexer->rec->state->tokenStartCharIndex,
									lexer->getCharIndex(lexer) - lexer->input->charByteSize
							);

}

static void *				
getCurrentInputSymbol		(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM istream)
{
	return NULL;
}

static void *				
getMissingSymbol			(pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM	istream, pANTLR3_EXCEPTION	e,
									ANTLR3_UINT32 expectedTokenType, pANTLR3_BITSET_LIST follow)
{
	return NULL;
}
