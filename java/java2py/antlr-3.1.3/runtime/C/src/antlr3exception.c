/** \file
 * Contains default functions for creating and destroying as well as
 * otherwise handling ANTLR3 standard exception structures.
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

#include    <antlr3exception.h>

static    void	antlr3ExceptionPrint(pANTLR3_EXCEPTION ex);
static    void	antlr3ExceptionFree (pANTLR3_EXCEPTION ex);

/**
 * \brief
 * Creates a new ANTLR3 exception structure
 * 
 * \param[in] exception
 * One of the ANTLR3_xxx_EXCEPTION indicators such as #ANTLR3_RECOGNITION_EXCEPTION
 * 
 * \param[in] message
 * Pointer to message string 
 * 
 * \param[in] freeMessage
 * Set to ANTLR3_TRUE if the message parameter should be freed by a call to 
 * ANTLR3_FREE() when the exception is destroyed.
 * 
 * \returns
 * Pointer to newly initialized exception structure, or an ANTLR3_ERR_xx defined value
 * upon failure.
 * 
 * An exception is 'thrown' by a recognizer  when input is seen that is not predicted by
 * the grammar productions or when some other error condition occurs. In C we do not have
 * the luxury of try and catch blocks, so exceptions are added in the order they occur to 
 * a list in the baserecognizer structure. The last one to be thrown is inserted at the head of
 * the list and the one currently installed is pointed to by the newly installed exception.
 * 
 * \remarks
 * After an exception is created, you may add a pointer to your own structure and a pointer
 * to a function to free this structure when the exception is destroyed.
 * 
 * \see
 * ANTLR3_EXCEPTION
 */
pANTLR3_EXCEPTION
antlr3ExceptionNew(ANTLR3_UINT32 exception, void * name, void * message, ANTLR3_BOOLEAN freeMessage)
{
	pANTLR3_EXCEPTION	ex;

	/* Allocate memory for the structure
	*/
	ex	= (pANTLR3_EXCEPTION) ANTLR3_CALLOC(1, sizeof(ANTLR3_EXCEPTION));

	/* Check for memory allocation
	*/
	if	(ex == NULL)
	{
		return	NULL;
	}

	ex->name		= name;		/* Install exception name	*/
	ex->type		= exception;	/* Install the exception number	*/
	ex->message		= message;	/* Install message string	*/

	/* Indicate whether the string should be freed if exception is destroyed    
	*/
	ex->freeMessage	= freeMessage;

	/* Install the API
	*/
	ex->print	    =  antlr3ExceptionPrint;
	ex->freeEx	    =  antlr3ExceptionFree;

	return ex;
}

/**
 * \brief
 * Prints out the message in all the exceptions in the supplied chain.
 * 
 * \param[in] ex
 * Pointer to the exception structure to print.
 * 
 * \remarks
 * You may wish to override this function by installing a pointer to a new function
 * in the base recognizer context structure.
 * 
 * \see
 * ANTLR3_BASE_RECOGNIZER
 */
static void
antlr3ExceptionPrint(pANTLR3_EXCEPTION ex)
{
    /* Ensure valid pointer
     */
    while   (ex != NULL)
    {
	/* Number if no message, else the message
	 */
	if  (ex->message == NULL)
	{
	    ANTLR3_FPRINTF(stderr, "ANTLR3_EXCEPTION number %d (%08X).\n", ex->type, ex->type);
	}
	else
	{
	    ANTLR3_FPRINTF(stderr, "ANTLR3_EXCEPTION: %s\n", (char *)(ex->message));
	}

	/* Move to next in the chain (if any)
	 */
	ex = ex->nextException;
    }

    return;
}

/**
 * \brief
 * Frees up a chain of ANTLR3 exceptions
 * 
 * \param[in] ex
 * Pointer to the first exception in the chain to free.
 * 
 * \see
 * ANTLR3_EXCEPTION
 */
static void
antlr3ExceptionFree(pANTLR3_EXCEPTION ex)
{
    pANTLR3_EXCEPTION next;

    /* Ensure valid pointer
     */
    while   (ex != NULL)
    {
	/* Pick up anythign following now, before we free the
	 * current memory block.
	 */
	next	= ex->nextException;

	/* Free the message pointer if advised to
	 */
	if  (ex->freeMessage == ANTLR3_TRUE)
	{
	    ANTLR3_FREE(ex->message);
	}

	/* Call the programmer's custom free routine if advised to
	 */
	if  (ex->freeCustom != NULL)
	{
	    ex->freeCustom(ex->custom);
	}

	/* Free the actual structure itself
	 */
	ANTLR3_FREE(ex);

	ex = next;
    }

    return;
}

