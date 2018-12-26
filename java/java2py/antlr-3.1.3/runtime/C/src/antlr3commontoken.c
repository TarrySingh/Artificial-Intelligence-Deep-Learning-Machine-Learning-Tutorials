/**
 * Contains the default implementation of the common token used within
 * java. Custom tokens should create this structure and then append to it using the 
 * custom pointer to install their own structure and API.
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

#include    <antlr3.h>

/* Token API
 */
static  pANTLR3_STRING	getText					(pANTLR3_COMMON_TOKEN token);
static  void			setText					(pANTLR3_COMMON_TOKEN token, pANTLR3_STRING text);
static  void			setText8				(pANTLR3_COMMON_TOKEN token, pANTLR3_UINT8 text);
static	ANTLR3_UINT32   getType					(pANTLR3_COMMON_TOKEN token);
static  void			setType					(pANTLR3_COMMON_TOKEN token, ANTLR3_UINT32 type);
static  ANTLR3_UINT32   getLine					(pANTLR3_COMMON_TOKEN token);
static  void			setLine					(pANTLR3_COMMON_TOKEN token, ANTLR3_UINT32 line);
static  ANTLR3_INT32    getCharPositionInLine	(pANTLR3_COMMON_TOKEN token);
static  void			setCharPositionInLine	(pANTLR3_COMMON_TOKEN token, ANTLR3_INT32 pos);
static  ANTLR3_UINT32   getChannel				(pANTLR3_COMMON_TOKEN token);
static  void			setChannel				(pANTLR3_COMMON_TOKEN token, ANTLR3_UINT32 channel);
static  ANTLR3_MARKER   getTokenIndex			(pANTLR3_COMMON_TOKEN token);
static  void			setTokenIndex			(pANTLR3_COMMON_TOKEN token, ANTLR3_MARKER);
static  ANTLR3_MARKER   getStartIndex			(pANTLR3_COMMON_TOKEN token);
static  void			setStartIndex			(pANTLR3_COMMON_TOKEN token, ANTLR3_MARKER index);
static  ANTLR3_MARKER   getStopIndex			(pANTLR3_COMMON_TOKEN token);
static  void			setStopIndex			(pANTLR3_COMMON_TOKEN token, ANTLR3_MARKER index);
static  pANTLR3_STRING  toString				(pANTLR3_COMMON_TOKEN token);

/* Factory API
 */
static	void			factoryClose	(pANTLR3_TOKEN_FACTORY factory);
static	pANTLR3_COMMON_TOKEN	newToken	(void);
static  void			setInputStream	(pANTLR3_TOKEN_FACTORY factory, pANTLR3_INPUT_STREAM input);

/* Internal management functions
 */
static	void			newPool		(pANTLR3_TOKEN_FACTORY factory);
static	pANTLR3_COMMON_TOKEN    newPoolToken	(pANTLR3_TOKEN_FACTORY factory);



ANTLR3_API pANTLR3_COMMON_TOKEN
antlr3CommonTokenNew(ANTLR3_UINT32 ttype)
{
	pANTLR3_COMMON_TOKEN    token;

	// Create a raw token with the interface installed
	//
	token   = newToken();

	if	(token != NULL)
	{
		token->setType(token, ttype);
	}

	// All good
	//
	return  token;
}

ANTLR3_API pANTLR3_TOKEN_FACTORY
antlr3TokenFactoryNew(pANTLR3_INPUT_STREAM input)
{
    pANTLR3_TOKEN_FACTORY   factory;

    /* allocate memory
     */
    factory	= (pANTLR3_TOKEN_FACTORY) ANTLR3_MALLOC((size_t)sizeof(ANTLR3_TOKEN_FACTORY));

    if	(factory == NULL)
    {
	return	NULL;
    }

    /* Install factory API
     */
    factory->newToken	    =  newPoolToken;
    factory->close			=  factoryClose;
    factory->setInputStream = setInputStream;
    
    /* Allocate the initial pool
     */
    factory->thisPool	= -1;
    factory->pools      = NULL;
    newPool(factory);

    /* Factory space is good, we now want to initialize our cheating token
     * which one it is initialized is the model for all tokens we manufacture
     */
    antlr3SetTokenAPI(&factory->unTruc);

    /* Set some initial variables for future copying
     */
    factory->unTruc.factoryMade	= ANTLR3_TRUE;

    // Input stream
    //
    setInputStream(factory, input);
    
    return  factory;

}

static void
setInputStream	(pANTLR3_TOKEN_FACTORY factory, pANTLR3_INPUT_STREAM input)
{
    factory->input          =  input;
    factory->unTruc.input   =  input;
	if	(input != NULL)
	{
		factory->unTruc.strFactory	= input->strFactory;
	}
	else
	{
		factory->unTruc.strFactory = NULL;
    }
}

static void
newPool(pANTLR3_TOKEN_FACTORY factory)
{
    /* Increment factory count
     */
    factory->thisPool++;

    /* Ensure we have enough pointers allocated
     */
    factory->pools = (pANTLR3_COMMON_TOKEN *)
		     ANTLR3_REALLOC(	(void *)factory->pools,	    /* Current pools pointer (starts at NULL)	*/
					(ANTLR3_UINT32)((factory->thisPool + 1) * sizeof(pANTLR3_COMMON_TOKEN *))	/* Memory for new pool pointers */
					);

    /* Allocate a new pool for the factory
     */
    factory->pools[factory->thisPool]	=
			    (pANTLR3_COMMON_TOKEN) 
				ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_COMMON_TOKEN) * ANTLR3_FACTORY_POOL_SIZE));

    /* Reset the counters
     */
    factory->nextToken	= 0;
  
    /* Done
     */
    return;
}

static pANTLR3_COMMON_TOKEN
newPoolToken(pANTLR3_TOKEN_FACTORY factory)
{
    pANTLR3_COMMON_TOKEN token;

    /* See if we need a new token pool before allocating a new
     * one
     */
    if (factory->nextToken >= ANTLR3_FACTORY_POOL_SIZE)
    {
        /* We ran out of tokens in the current pool, so we need a new pool
         */
        newPool(factory);
    }

    /* Assuming everything went well (we are trying for performance here so doing minimal
     * error checking. Then we can work out what the pointer is to the next token.
     */
    token = factory->pools[factory->thisPool] + factory->nextToken;
    factory->nextToken++;

    /* We have our token pointer now, so we can initialize it to the predefined model.
     */
    antlr3SetTokenAPI(token);

    /* It is factory made, and we need to copy the string factory pointer
     */
    token->factoryMade  = ANTLR3_TRUE;
    token->strFactory   = factory->input == NULL ? NULL : factory->input->strFactory;
    token->input        = factory->input;

    /* And we are done
     */
    return token;
}

static	void
factoryClose	    (pANTLR3_TOKEN_FACTORY factory)
{
    pANTLR3_COMMON_TOKEN    pool;
    ANTLR3_INT32	    poolCount;
    ANTLR3_UINT32	    limit;
    ANTLR3_UINT32	    token;
    pANTLR3_COMMON_TOKEN    check;

    /* We iterate the token pools one at a time
     */
    for	(poolCount = 0; poolCount <= factory->thisPool; poolCount++)
    {
	/* Pointer to current pool
	 */
	pool	= factory->pools[poolCount];

	/* Work out how many tokens we need to check in this pool.
	 */
	limit	= (poolCount == factory->thisPool ? factory->nextToken : ANTLR3_FACTORY_POOL_SIZE);
	
	/* Marginal condition, we might be at the start of a brand new pool
	 * where the nextToken is 0 and nothing has been allocated.
	 */
	if  (limit > 0)
	{
	    /* We have some tokens allocated from this pool
	     */
	    for (token = 0; token < limit; token++)
	    {
		/* Next one in the chain
		 */
		check	= pool + token;

		/* If the programmer made this a custom token, then
		 * see if we need to call their free routine.
		 */
		if  (check->custom != NULL && check->freeCustom != NULL)
		{
		    check->freeCustom(check->custom);
		    check->custom = NULL;
		}
	    }
	}

	/* We can now free this pool allocation
	 */
	ANTLR3_FREE(factory->pools[poolCount]);
	factory->pools[poolCount] = NULL;
    }

    /* All the pools are deallocated we can free the pointers to the pools
     * now.
     */
    ANTLR3_FREE(factory->pools);

    /* Finally, we can free the space for the factory itself
     */
    ANTLR3_FREE(factory);
}


static	pANTLR3_COMMON_TOKEN	
newToken(void)
{
    pANTLR3_COMMON_TOKEN    token;

    /* Allocate memory for this
     */
    token   = (pANTLR3_COMMON_TOKEN) ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_COMMON_TOKEN)));

    if	(token == NULL)
    {
	return	NULL;
    }

    // Install the API
    //
    antlr3SetTokenAPI(token);
    token->factoryMade = ANTLR3_FALSE;

    return  token;
}

ANTLR3_API void
antlr3SetTokenAPI(pANTLR3_COMMON_TOKEN token)
{
    token->getText					= getText;
    token->setText					= setText;
    token->setText8					= setText8;
    token->getType					= getType;
    token->setType					= setType;
    token->getLine					= getLine;
    token->setLine				    = setLine;
    token->setLine					= setLine;
    token->getCharPositionInLine    = getCharPositionInLine;
    token->setCharPositionInLine    = setCharPositionInLine;
    token->getChannel				= getChannel;
    token->setChannel				= setChannel;
    token->getTokenIndex			= getTokenIndex;
    token->setTokenIndex			= setTokenIndex;
    token->getStartIndex			= getStartIndex;
    token->setStartIndex			= setStartIndex;
    token->getStopIndex				= getStopIndex;
    token->setStopIndex				= setStopIndex;
    token->toString					= toString;

    // Set defaults
    //
    token->setCharPositionInLine(token, -1);

    token->custom		    = NULL;
    token->freeCustom	    = NULL;
    token->type			    = ANTLR3_TOKEN_INVALID;
	token->textState		= ANTLR3_TEXT_NONE;
    token->start		    = 0;
    token->stop			    = 0;
    token->channel		    = ANTLR3_TOKEN_DEFAULT_CHANNEL;
    token->line			    = 0;
    token->index		    = 0;
    token->input		    = NULL;
	token->user1			= 0;
	token->user2			= 0;
	token->user3			= 0;
	token->custom			= NULL;

    return;
}

static  pANTLR3_STRING  getText			(pANTLR3_COMMON_TOKEN token)
{
	switch (token->textState)
	{
		case ANTLR3_TEXT_STRING:

			// Someone already created a string for this token, so we just
			// use it.
			//
			return	token->tokText.text;
			break;
    
		case ANTLR3_TEXT_CHARP:

			// We had a straight text pointer installed, now we
			// must convert it to a string. Note we have to do this here
			// or otherwise setText8() will just install the same char*
			//
			if	(token->strFactory != NULL)
			{
				token->tokText.text	= token->strFactory->newStr8(token->strFactory, (pANTLR3_UINT8)token->tokText.chars);
				token->textState	= ANTLR3_TEXT_STRING;
				return token->tokText.text;
			}
			else
			{
				// We cannot do anything here
				//
				return NULL;
			}
			break;

		default:

			// EOF is a special case
			//
			if (token->type == ANTLR3_TOKEN_EOF)
			{
				token->tokText.text	= token->strFactory->newStr8(token->strFactory, (pANTLR3_UINT8)"<EOF>");
				token->textState	= ANTLR3_TEXT_STRING;
				return token->tokText.text;
			}


			// We had nothing installed in the token, create a new string
			// from the input stream
			//

			if	(token->input != NULL)
			{
			
				return	token->input->substr(	token->input, 
												token->getStartIndex(token), 
 												token->getStopIndex(token)
											);
			}

			// Nothing to return, there is no input stream
			//
			return NULL;
			break;
	}
}
static  void		setText8		(pANTLR3_COMMON_TOKEN token, pANTLR3_UINT8 text)
{
	// No text to set, so ignore
	//
	if	(text == NULL) return;

	switch	(token->textState)
	{
		case	ANTLR3_TEXT_NONE:
		case	ANTLR3_TEXT_CHARP:	// Caller must free before setting again, if it needs to be freed

			// Nothing in there yet, or just a char *, so just set the
			// text as a pointer
			//
			token->textState		= ANTLR3_TEXT_CHARP;
			token->tokText.chars	= (pANTLR3_UCHAR)text;
			break;

		default:

			// It was already a pANTLR3_STRING, so just override it
			//
			token->tokText.text->set8(token->tokText.text, (const char *)text);
			break;
	}

	// We are done 
	//
	return;
}

/** \brief Install the supplied text string as teh text for the token.
 * The method assumes that the existing text (if any) was created by a factory
 * and so does not attempt to release any memory it is using.Text not created
 * by a string fctory (not advised) should be released prior to this call.
 */
static  void		setText			(pANTLR3_COMMON_TOKEN token, pANTLR3_STRING text)
{
	// Merely replaces and existing pre-defined text with the supplied
	// string
	//
	token->textState	= ANTLR3_TEXT_STRING;
	token->tokText.text	= text;

	/* We are done 
	*/
	return;
}

static	ANTLR3_UINT32   getType			(pANTLR3_COMMON_TOKEN token)
{
    return  token->type;
}

static  void		setType			(pANTLR3_COMMON_TOKEN token, ANTLR3_UINT32 type)
{
    token->type = type;
}

static  ANTLR3_UINT32   getLine			(pANTLR3_COMMON_TOKEN token)
{
    return  token->line;
}

static  void		setLine			(pANTLR3_COMMON_TOKEN token, ANTLR3_UINT32 line)
{
    token->line = line;
}

static  ANTLR3_INT32    getCharPositionInLine	(pANTLR3_COMMON_TOKEN token)
{
    return  token->charPosition;
}

static  void		setCharPositionInLine	(pANTLR3_COMMON_TOKEN token, ANTLR3_INT32 pos)
{
    token->charPosition = pos;
}

static  ANTLR3_UINT32   getChannel		(pANTLR3_COMMON_TOKEN token)
{
    return  token->channel;
}

static  void		setChannel		(pANTLR3_COMMON_TOKEN token, ANTLR3_UINT32 channel)
{
    token->channel  = channel;
}

static  ANTLR3_MARKER   getTokenIndex		(pANTLR3_COMMON_TOKEN token)
{
    return  token->index;
}

static  void		setTokenIndex		(pANTLR3_COMMON_TOKEN token, ANTLR3_MARKER index)
{
    token->index    = index;
}

static  ANTLR3_MARKER   getStartIndex		(pANTLR3_COMMON_TOKEN token)
{
	return  token->start == -1 ? (ANTLR3_MARKER)(token->input->data) : token->start;
}

static  void		setStartIndex		(pANTLR3_COMMON_TOKEN token, ANTLR3_MARKER start)
{
    token->start    = start;
}

static  ANTLR3_MARKER   getStopIndex		(pANTLR3_COMMON_TOKEN token)
{
    return  token->stop;
}

static  void		setStopIndex		(pANTLR3_COMMON_TOKEN token, ANTLR3_MARKER stop)
{
    token->stop	= stop;
}

static  pANTLR3_STRING    toString		(pANTLR3_COMMON_TOKEN token)
{
    pANTLR3_STRING  text;
    pANTLR3_STRING  outtext;

    text    =	token->getText(token);
    
    if	(text == NULL)
    {
		return NULL;
    }

	if	(text->factory == NULL)
	{
		return text;		// This usall ymeans it is the EOF token
	}

    /* A new empty string to assemble all the stuff in
     */
    outtext = text->factory->newRaw(text->factory);

    /* Now we use our handy dandy string utility to assemble the
     * the reporting string
     * return "[@"+getTokenIndex()+","+start+":"+stop+"='"+txt+"',<"+type+">"+channelStr+","+line+":"+getCharPositionInLine()+"]";
     */
    outtext->append8(outtext, "[Index: ");
    outtext->addi   (outtext, (ANTLR3_INT32)token->getTokenIndex(token));
    outtext->append8(outtext, " (Start: ");
    outtext->addi   (outtext, (ANTLR3_INT32)token->getStartIndex(token));
    outtext->append8(outtext, "-Stop: ");
    outtext->addi   (outtext, (ANTLR3_INT32)token->getStopIndex(token));
    outtext->append8(outtext, ") ='");
    outtext->appendS(outtext, text);
    outtext->append8(outtext, "', type<");
    outtext->addi   (outtext, token->type);
    outtext->append8(outtext, "> ");

    if	(token->getChannel(token) > ANTLR3_TOKEN_DEFAULT_CHANNEL)
    {
	outtext->append8(outtext, "(channel = ");
	outtext->addi	(outtext, (ANTLR3_INT32)token->getChannel(token));
	outtext->append8(outtext, ") ");
    }

    outtext->append8(outtext, "Line: ");
    outtext->addi   (outtext, (ANTLR3_INT32)token->getLine(token));
    outtext->append8(outtext, " LinePos:");
    outtext->addi   (outtext, token->getCharPositionInLine(token));
    outtext->addc   (outtext, ']');

    return  outtext;
}

