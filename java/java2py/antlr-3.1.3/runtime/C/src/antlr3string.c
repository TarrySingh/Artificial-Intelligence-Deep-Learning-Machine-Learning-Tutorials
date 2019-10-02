/** \file
 * Implementation of the ANTLR3 string and string factory classes
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

#include    <antlr3string.h>

/* Factory API
 */
static    pANTLR3_STRING    newRaw8	(pANTLR3_STRING_FACTORY factory);
static    pANTLR3_STRING    newRaw16	(pANTLR3_STRING_FACTORY factory);
static    pANTLR3_STRING    newSize8	(pANTLR3_STRING_FACTORY factory, ANTLR3_UINT32 size);
static    pANTLR3_STRING    newSize16	(pANTLR3_STRING_FACTORY factory, ANTLR3_UINT32 size);
static    pANTLR3_STRING    newPtr8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 string, ANTLR3_UINT32 size);
static    pANTLR3_STRING    newPtr16_8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 string, ANTLR3_UINT32 size);
static    pANTLR3_STRING    newPtr16_16	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 string, ANTLR3_UINT32 size);
static    pANTLR3_STRING    newStr8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 string);
static    pANTLR3_STRING    newStr16_8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 string);
static    pANTLR3_STRING    newStr16_16	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 string);
static    void		    destroy	(pANTLR3_STRING_FACTORY factory, pANTLR3_STRING string);
static    pANTLR3_STRING    printable8	(pANTLR3_STRING_FACTORY factory, pANTLR3_STRING string);
static    pANTLR3_STRING    printable16	(pANTLR3_STRING_FACTORY factory, pANTLR3_STRING string);
static    void		    closeFactory(pANTLR3_STRING_FACTORY factory);

/* String API
 */
static    pANTLR3_UINT8	    set8	(pANTLR3_STRING string, const char * chars);
static    pANTLR3_UINT8	    set16_8	(pANTLR3_STRING string, const char * chars);
static    pANTLR3_UINT8	    set16_16	(pANTLR3_STRING string, const char * chars);
static    pANTLR3_UINT8	    append8	(pANTLR3_STRING string, const char * newbit);
static    pANTLR3_UINT8	    append16_8	(pANTLR3_STRING string, const char * newbit);
static    pANTLR3_UINT8	    append16_16	(pANTLR3_STRING string, const char * newbit);
static	  pANTLR3_UINT8	    insert8	(pANTLR3_STRING string, ANTLR3_UINT32 point, const char * newbit);
static	  pANTLR3_UINT8	    insert16_8	(pANTLR3_STRING string, ANTLR3_UINT32 point, const char * newbit);
static	  pANTLR3_UINT8	    insert16_16	(pANTLR3_STRING string, ANTLR3_UINT32 point, const char * newbit);

static    pANTLR3_UINT8	    setS	(pANTLR3_STRING string, pANTLR3_STRING chars);
static    pANTLR3_UINT8	    appendS	(pANTLR3_STRING string, pANTLR3_STRING newbit);
static	  pANTLR3_UINT8	    insertS	(pANTLR3_STRING string, ANTLR3_UINT32 point, pANTLR3_STRING newbit);

static    pANTLR3_UINT8	    addc8	(pANTLR3_STRING string, ANTLR3_UINT32 c);
static    pANTLR3_UINT8	    addc16	(pANTLR3_STRING string, ANTLR3_UINT32 c);
static    pANTLR3_UINT8	    addi8	(pANTLR3_STRING string, ANTLR3_INT32 i);
static    pANTLR3_UINT8	    addi16	(pANTLR3_STRING string, ANTLR3_INT32 i);
static	  pANTLR3_UINT8	    inserti8	(pANTLR3_STRING string, ANTLR3_UINT32 point, ANTLR3_INT32 i);
static	  pANTLR3_UINT8	    inserti16	(pANTLR3_STRING string, ANTLR3_UINT32 point, ANTLR3_INT32 i);

static    ANTLR3_UINT32     compare8	(pANTLR3_STRING string, const char * compStr);
static    ANTLR3_UINT32     compare16_8	(pANTLR3_STRING string, const char * compStr);
static    ANTLR3_UINT32     compare16_16(pANTLR3_STRING string, const char * compStr);
static    ANTLR3_UINT32     compareS	(pANTLR3_STRING string, pANTLR3_STRING compStr);
static    ANTLR3_UCHAR      charAt8	(pANTLR3_STRING string, ANTLR3_UINT32 offset);
static    ANTLR3_UCHAR      charAt16	(pANTLR3_STRING string, ANTLR3_UINT32 offset);
static    pANTLR3_STRING    subString8	(pANTLR3_STRING string, ANTLR3_UINT32 startIndex, ANTLR3_UINT32 endIndex);
static    pANTLR3_STRING    subString16	(pANTLR3_STRING string, ANTLR3_UINT32 startIndex, ANTLR3_UINT32 endIndex);
static	  ANTLR3_INT32	    toInt32_8	(pANTLR3_STRING string);
static	  ANTLR3_INT32	    toInt32_16  (pANTLR3_STRING string);
static	  pANTLR3_STRING    to8_8		(pANTLR3_STRING string);
static	  pANTLR3_STRING    to8_16		(pANTLR3_STRING string);
static	pANTLR3_STRING		toUTF8_8	(pANTLR3_STRING string);
static	pANTLR3_STRING		toUTF8_16	(pANTLR3_STRING string);

/* Local helpers
 */
static	void			stringInit8	(pANTLR3_STRING string);
static	void			stringInit16	(pANTLR3_STRING string);
static	void	ANTLR3_CDECL	stringFree	(pANTLR3_STRING string);

ANTLR3_API pANTLR3_STRING_FACTORY 
antlr3StringFactoryNew()
{
	pANTLR3_STRING_FACTORY  factory;

	/* Allocate memory
	*/
	factory	= (pANTLR3_STRING_FACTORY) ANTLR3_MALLOC(sizeof(ANTLR3_STRING_FACTORY));

	if	(factory == NULL)
	{
		return	NULL;
	}

	/* Now we make a new list to track the strings.
	*/
	factory->strings	= antlr3VectorNew(0);
	factory->index	= 0;

	if	(factory->strings == NULL)
	{
		ANTLR3_FREE(factory);
		return	NULL;
	}

	/* Install the API (8 bit assumed)
	*/
	factory->newRaw		=  newRaw8;
	factory->newSize	=  newSize8;

	factory->newPtr		=  newPtr8;
	factory->newPtr8	=  newPtr8;
	factory->newStr		=  newStr8;
	factory->newStr8	=  newStr8;
	factory->destroy	=  destroy;
	factory->printable	=  printable8;
	factory->destroy	=  destroy;
	factory->close		=  closeFactory;

	return  factory;
}

/** Create a string factory that is UCS2 (16 bit) encoding based
 */
ANTLR3_API pANTLR3_STRING_FACTORY 
antlr3UCS2StringFactoryNew()
{
     pANTLR3_STRING_FACTORY  factory;

    /* Allocate an 8 bit factory, then override with 16 bit UCS2 functions where we 
     * need to.
     */
    factory	= antlr3StringFactoryNew();

    if	(factory == NULL)
    {
		return	NULL;
    }

    /* Override the 8 bit API with the UCS2 (mostly just 16 bit) API
     */
    factory->newRaw	=  newRaw16;
    factory->newSize	=  newSize16;

    factory->newPtr	=  newPtr16_16;
    factory->newPtr8	=  newPtr16_8;
    factory->newStr	=  newStr16_16;
    factory->newStr8	=  newStr16_8;
    factory->printable	=  printable16;

    factory->destroy	=  destroy;
    factory->destroy	=  destroy;
    factory->close	=  closeFactory;

    return  factory;
}

/**
 *
 * \param factory 
 * \return 
 */
static    pANTLR3_STRING    
newRaw8	(pANTLR3_STRING_FACTORY factory)
{
    pANTLR3_STRING  string;

    string  = (pANTLR3_STRING) ANTLR3_MALLOC(sizeof(ANTLR3_STRING));

    if	(string == NULL)
    {
		return	NULL;
    }

    /* Structure is allocated, now fill in the API etc.
     */
    stringInit8(string);
    string->factory = factory;

    /* Add the string into the allocated list
     */
    factory->strings->set(factory->strings, factory->index, (void *) string, (void (ANTLR3_CDECL *)(void *))(stringFree), ANTLR3_TRUE);
    string->index   = factory->index++;

    return string;
}
/**
 *
 * \param factory 
 * \return 
 */
static    pANTLR3_STRING    
newRaw16	(pANTLR3_STRING_FACTORY factory)
{
    pANTLR3_STRING  string;

    string  = (pANTLR3_STRING) ANTLR3_MALLOC(sizeof(ANTLR3_STRING));

    if	(string == NULL)
    {
		return	NULL;
    }

    /* Structure is allocated, now fill in the API etc.
     */
    stringInit16(string);
    string->factory = factory;

    /* Add the string into the allocated list
     */
    factory->strings->set(factory->strings, factory->index, (void *) string, (void (ANTLR3_CDECL *)(void *))(stringFree), ANTLR3_TRUE);
    string->index   = factory->index++;

    return string;
}
static	 
void	ANTLR3_CDECL stringFree  (pANTLR3_STRING string)
{
    /* First free the string itself if there was anything in it
     */
    if	(string->chars)
    {
	ANTLR3_FREE(string->chars);
    }

    /* Now free the space for this string
     */
    ANTLR3_FREE(string);

    return;
}
/**
 *
 * \param string 
 * \return 
 */
static	void
stringInit8  (pANTLR3_STRING string)
{
    string->len			= 0;
    string->size		= 0;
    string->chars		= NULL;
    string->encoding	= ANTLR3_ENCODING_LATIN1;

    /* API for 8 bit strings*/

    string->set		= set8;
    string->set8	= set8;
    string->append	= append8;
    string->append8	= append8;
    string->insert	= insert8;
    string->insert8	= insert8;
    string->addi	= addi8;
    string->inserti	= inserti8;
    string->addc	= addc8;
    string->charAt	= charAt8;
    string->compare	= compare8;
    string->compare8	= compare8;
    string->subString	= subString8;
    string->toInt32	= toInt32_8;
    string->to8		= to8_8;
	string->toUTF8	= toUTF8_8;
    string->compareS	= compareS;
    string->setS	= setS;
    string->appendS	= appendS;
    string->insertS	= insertS;

}
/**
 *
 * \param string 
 * \return 
 */
static	void
stringInit16  (pANTLR3_STRING string)
{
    string->len		= 0;
    string->size	= 0;
    string->chars	= NULL;
    string->encoding	= ANTLR3_ENCODING_UCS2;

    /* API for 16 bit strings */

    string->set		= set16_16;
    string->set8	= set16_8;
    string->append	= append16_16;
    string->append8	= append16_8;
    string->insert	= insert16_16;
    string->insert8	= insert16_8;
    string->addi	= addi16;
    string->inserti	= inserti16;
    string->addc	= addc16;
    string->charAt	= charAt16;
    string->compare	= compare16_16;
    string->compare8	= compare16_8;
    string->subString	= subString16;
    string->toInt32	= toInt32_16;
    string->to8		= to8_16;
	string->toUTF8	= toUTF8_16;

    string->compareS	= compareS;
    string->setS	= setS;
    string->appendS	= appendS;
    string->insertS	= insertS;
}
/**
 *
 * \param string 
 * \return 
 * TODO: Implement UTF-8
 */
static	void
stringInitUTF8  (pANTLR3_STRING string)
{
    string->len	    = 0;
    string->size    = 0;
    string->chars   = NULL;

    /* API */

}

// Convert an 8 bit string into a UTF8 representation, which is in fact just the string itself
// a memcpy as we make no assumptions about the 8 bit encoding.
//
static	pANTLR3_STRING		
toUTF8_8	(pANTLR3_STRING string)
{
	return string->factory->newPtr(string->factory, (pANTLR3_UINT8)(string->chars), string->len);
}

// Convert a 16 bit (UCS2) string into a UTF8 representation using the Unicode.org
// supplied C algorithms, which are now contained within the ANTLR3 C runtime
// as permitted by the Unicode license (within the source code antlr3convertutf.c/.h
// UCS2 has the same encoding as UTF16 so we can use UTF16 converter.
//
static	pANTLR3_STRING	
toUTF8_16	(pANTLR3_STRING string)
{
	
	UTF8				* outputEnd;	
	UTF16				* inputEnd;
	pANTLR3_STRING		utf8String;

	ConversionResult	cResult;

	// Allocate the output buffer, which needs to accommodate potentially
	// 3X (in bytes) the input size (in chars).
	//
	utf8String	= string->factory->newStr8(string->factory, (pANTLR3_UINT8)"");

	if	(utf8String != NULL)
	{
		// Free existing allocation
		//
		ANTLR3_FREE(utf8String->chars);

		// Reallocate according to maximum expected size
		//
		utf8String->size	= string->len *3;
		utf8String->chars	= (pANTLR3_UINT8)ANTLR3_MALLOC(utf8String->size +1);

		if	(utf8String->chars != NULL)
		{
			inputEnd  = (UTF16 *)	(string->chars);
			outputEnd = (UTF8 *)	(utf8String->chars);

			// Call the Unicode converter
			//
			cResult =  ConvertUTF16toUTF8
							(
								(const UTF16**)&inputEnd, 
								((const UTF16 *)(string->chars)) + string->len, 
								&outputEnd, 
								outputEnd + utf8String->size - 1,
								lenientConversion
							);

			// We don't really care if things failed or not here, we just converted
			// everything that was vaguely possible and stopped when it wasn't. It is
			// up to the grammar programmer to verify that the input is sensible.
			//
			utf8String->len = ANTLR3_UINT32_CAST(((pANTLR3_UINT8)outputEnd) - utf8String->chars);

			*(outputEnd+1) = '\0';		// Always null terminate
		}
	}
	return utf8String;
}

/**
 * Creates a new string with enough capacity for size 8 bit characters plus a terminator.
 *
 * \param[in] factory - Pointer to the string factory that owns strings
 * \param[in] size - In characters
 * \return pointer to the new string.
 */
static    pANTLR3_STRING    
newSize8	(pANTLR3_STRING_FACTORY factory, ANTLR3_UINT32 size)
{
    pANTLR3_STRING  string;

    string  = factory->newRaw(factory);

    if	(string == NULL)
    {
		return	string;
    }

    /* Always add one more byte for a terminator ;-)
     */
    string->chars		= (pANTLR3_UINT8) ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_UINT8) * (size+1)));
	*(string->chars)	= '\0';
    string->size		= size + 1;


    return string;
}
/**
 * Creates a new string with enough capacity for size 16 bit characters plus a terminator.
 *
 * \param[in] factory - POitner to the string factory that owns strings
 * \param[in] size - In characters
 * \return pointer to the new string.
 */
static    pANTLR3_STRING    
newSize16	(pANTLR3_STRING_FACTORY factory, ANTLR3_UINT32 size)
{
    pANTLR3_STRING  string;

    string  = factory->newRaw(factory);

    if	(string == NULL)
    {
		return	string;
    }

    /* Always add one more byte for a terminator ;-)
     */	
    string->chars		= (pANTLR3_UINT8) ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_UINT16) * (size+1)));
	*(string->chars)	= '\0';
    string->size		= size+1;	/* Size is always in characters, as is len */

    return string;
}

/** Creates a new 8 bit string initialized with the 8 bit characters at the 
 *  supplied ptr, of pre-determined size.
 * \param[in] factory - Pointer to the string factory that owns the strings
 * \param[in] ptr - Pointer to 8 bit encoded characters
 * \return pointer to the new string
 */
static    pANTLR3_STRING    
newPtr8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 ptr, ANTLR3_UINT32 size)
{
	pANTLR3_STRING  string;

	string  = factory->newSize(factory, size);

	if	(string == NULL)
	{
		return	NULL;
	}

	if	(size <= 0)
	{
		return	string;
	}

	if	(ptr != NULL)
	{
		ANTLR3_MEMMOVE(string->chars, (const void *)ptr, size);
		*(string->chars + size) = '\0';	    /* Terminate, these strings are usually used for Token streams and printing etc.	*/
		string->len = size;
	}

	return  string;
}

/** Creates a new 16 bit string initialized with the 8 bit characters at the 
 *  supplied 8 bit character ptr, of pre-determined size.
 * \param[in] factory - Pointer to the string factory that owns the strings
 * \param[in] ptr - Pointer to 8 bit encoded characters
 * \return pointer to the new string
 */
static    pANTLR3_STRING    
newPtr16_8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 ptr, ANTLR3_UINT32 size)
{
	pANTLR3_STRING  string;

	/* newSize accepts size in characters, not bytes
	*/
	string  = factory->newSize(factory, size);

	if	(string == NULL)
	{
		return	NULL;
	}

	if	(size <= 0)
	{
		return	string;
	}

	if	(ptr != NULL)
	{
		pANTLR3_UINT16	out;
		ANTLR3_INT32    inSize;

		out = (pANTLR3_UINT16)(string->chars);
		inSize	= size;

		while	(inSize-- > 0)
		{
			*out++ = (ANTLR3_UINT16)(*ptr++);
		}

		/* Terminate, these strings are usually used for Token streams and printing etc.	
		*/
		*(((pANTLR3_UINT16)(string->chars)) + size) = '\0';

		string->len = size;
	}

	return  string;
}

/** Creates a new 16 bit string initialized with the 16 bit characters at the 
 *  supplied ptr, of pre-determined size.
 * \param[in] factory - Pointer to the string factory that owns the strings
 * \param[in] ptr - Pointer to 16 bit encoded characters
 * \return pointer to the new string
 */
static    pANTLR3_STRING    
newPtr16_16	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 ptr, ANTLR3_UINT32 size)
{
	pANTLR3_STRING  string;

	string  = factory->newSize(factory, size);

	if	(string == NULL)
	{
		return	NULL;
	}

	if	(size <= 0)
	{
		return	string;
	}

	if	(ptr != NULL)
	{
		ANTLR3_MEMMOVE(string->chars, (const void *)ptr, (size * sizeof(ANTLR3_UINT16)));

		/* Terminate, these strings are usually used for Token streams and printing etc.	
		*/
		*(((pANTLR3_UINT16)(string->chars)) + size) = '\0';	    
		string->len = size;
	}

	return  string;
}

/** Create a new 8 bit string from the supplied, null terminated, 8 bit string pointer.
 * \param[in] factory - Pointer to the string factory that owns strings.
 * \param[in] ptr - Pointer to the 8 bit encoded string
 * \return Pointer to the newly initialized string
 */
static    pANTLR3_STRING    
newStr8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 ptr)
{
    return factory->newPtr8(factory, ptr, (ANTLR3_UINT32)strlen((const char *)ptr));
}

/** Create a new 16 bit string from the supplied, null terminated, 8 bit string pointer.
 * \param[in] factory - Pointer to the string factory that owns strings.
 * \param[in] ptr - Pointer to the 8 bit encoded string
 * \return POinter to the newly initialized string
 */
static    pANTLR3_STRING    
newStr16_8	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 ptr)
{
    return factory->newPtr8(factory, ptr, (ANTLR3_UINT32)strlen((const char *)ptr));
}

/** Create a new 16 bit string from the supplied, null terminated, 16 bit string pointer.
 * \param[in] factory - Pointer to the string factory that owns strings.
 * \param[in] ptr - Pointer to the 16 bit encoded string
 * \return Pointer to the newly initialized string
 */
static    pANTLR3_STRING    
newStr16_16	(pANTLR3_STRING_FACTORY factory, pANTLR3_UINT8 ptr)
{
    pANTLR3_UINT16  in;
    ANTLR3_UINT32   count;

    /** First, determine the length of the input string
     */
    in	    = (pANTLR3_UINT16)ptr;
    count   = 0;

    while   (*in++ != '\0')
    {
		count++;
    }
    return factory->newPtr(factory, ptr, count);
}

static    void		    
destroy	(pANTLR3_STRING_FACTORY factory, pANTLR3_STRING string)
{
    // Record which string we are deleting
    //
    ANTLR3_UINT32 strIndex = string->index;
    
    // Ensure that the string was not factory made, or we would try
    // to delete memory that wasn't allocated outside the factory
    // block.
    // Remove the specific indexed string from the vector
    //
    factory->strings->del(factory->strings, strIndex);

    // One less string in the vector, so decrement the factory index
    // so that the next string allocated is indexed correctly with
    // respect to the vector.
    //
    factory->index--;

    // Now we have to reindex the strings in the vector that followed
    // the one we just deleted. We only do this if the one we just deleted
    // was not the last one.
    //
    if  (strIndex< factory->index)
    {
        // We must reindex the strings after the one we just deleted.
        // The one that follows the one we just deleted is also out
        // of whack, so we start there.
        //
        ANTLR3_UINT32 i;

        for (i = strIndex; i < factory->index; i++)
        {
            // Renumber the entry
            //
            ((pANTLR3_STRING)(factory->strings->elements[i].element))->index = i;
        }
    }

    // The string has been destroyed and the elements of the factory are reindexed.
    //

}

static    pANTLR3_STRING    
printable8(pANTLR3_STRING_FACTORY factory, pANTLR3_STRING instr)
{
    pANTLR3_STRING  string;
    
    /* We don't need to be too efficient here, this is mostly for error messages and so on.
     */
    pANTLR3_UINT8   scannedText;
    ANTLR3_UINT32   i;

    /* Assume we need as much as twice as much space to parse out the control characters
     */
    string  = factory->newSize(factory, instr->len *2 + 1);

    /* Scan through and replace unprintable (in terms of this routine)
     * characters
     */
    scannedText = string->chars;

    for	(i = 0; i < instr->len; i++)
    {
		if (*(instr->chars + i) == '\n')
		{
			*scannedText++ = '\\';
			*scannedText++ = 'n';
		}
		else if (*(instr->chars + i) == '\r')
		{
			*scannedText++ = '\\';
			*scannedText++ = 'r';
		}
		else if	(!isprint(*(instr->chars +i)))
		{
			*scannedText++ = '?';
		}
		else
		{
			*scannedText++ = *(instr->chars + i);
		}
    }
    *scannedText  = '\0';

    string->len	= (ANTLR3_UINT32)(scannedText - string->chars);
    
    return  string;
}

static    pANTLR3_STRING    
printable16(pANTLR3_STRING_FACTORY factory, pANTLR3_STRING instr)
{
    pANTLR3_STRING  string;
    
    /* We don't need to be too efficient here, this is mostly for error messages and so on.
     */
    pANTLR3_UINT16  scannedText;
    pANTLR3_UINT16  inText;
    ANTLR3_UINT32   i;
    ANTLR3_UINT32   outLen;

    /* Assume we need as much as twice as much space to parse out the control characters
     */
    string  = factory->newSize(factory, instr->len *2 + 1);

    /* Scan through and replace unprintable (in terms of this routine)
     * characters
     */
    scannedText = (pANTLR3_UINT16)(string->chars);
    inText	= (pANTLR3_UINT16)(instr->chars);
    outLen	= 0;

    for	(i = 0; i < instr->len; i++)
    {
		if (*(inText + i) == '\n')
		{
			*scannedText++   = '\\';
			*scannedText++   = 'n';
			outLen	    += 2;
		}
		else if (*(inText + i) == '\r')
		{
			*scannedText++   = '\\';
			*scannedText++   = 'r';
			outLen	    += 2;
		}
		else if	(!isprint(*(inText +i)))
		{
			*scannedText++ = '?';
			outLen++;
		}
		else
		{
			*scannedText++ = *(inText + i);
			outLen++;
		}
    }
    *scannedText  = '\0';

    string->len	= outLen;
    
    return  string;
}

/** Fascist Capitalist Pig function created
 *  to oppress the workers comrade.
 */
static    void		    
closeFactory	(pANTLR3_STRING_FACTORY factory)
{
    /* Delete the vector we were tracking the strings with, this will
     * causes all the allocated strings to be deallocated too
     */
    factory->strings->free(factory->strings);

    /* Delete the space for the factory itself
     */
    ANTLR3_FREE((void *)factory);
}

static    pANTLR3_UINT8   
append8	(pANTLR3_STRING string, const char * newbit)
{
    ANTLR3_UINT32 len;

    len	= (ANTLR3_UINT32)strlen(newbit);

    if	(string->size < (string->len + len + 1))
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(string->len + len + 1));
		string->size	= string->len + len + 1;
    }

    /* Note we copy one more byte than the strlen in order to get the trailing
     */
    ANTLR3_MEMMOVE((void *)(string->chars + string->len), newbit, (ANTLR3_UINT32)(len+1));
    string->len	+= len;

    return string->chars;
}

static    pANTLR3_UINT8   
append16_8	(pANTLR3_STRING string, const char * newbit)
{
    ANTLR3_UINT32   len;
    pANTLR3_UINT16  apPoint;
    ANTLR3_UINT32   count;

    len	= (ANTLR3_UINT32)strlen(newbit);

    if	(string->size < (string->len + len + 1))
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)((sizeof(ANTLR3_UINT16)*(string->len + len + 1))));
		string->size	= string->len + len + 1;
    }

    apPoint = ((pANTLR3_UINT16)string->chars) + string->len;
    string->len	+= len;

    for	(count = 0; count < len; count++)
    {
		*apPoint++   = *(newbit + count);
    }
    *apPoint = '\0';

    return string->chars;
}

static    pANTLR3_UINT8   
append16_16	(pANTLR3_STRING string, const char * newbit)
{
    ANTLR3_UINT32 len;
    pANTLR3_UINT16  in;

    /** First, determine the length of the input string
     */
    in	    = (pANTLR3_UINT16)newbit;
    len   = 0;

    while   (*in++ != '\0')
    {
		len++;
    }

    if	(string->size < (string->len + len + 1))
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)( sizeof(ANTLR3_UINT16) *(string->len + len + 1) ));
		string->size	= string->len + len + 1;
    }

    /* Note we copy one more byte than the strlen in order to get the trailing delimiter
     */
    ANTLR3_MEMMOVE((void *)(((pANTLR3_UINT16)string->chars) + string->len), newbit, (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(len+1)));
    string->len	+= len;

    return string->chars;
}

static    pANTLR3_UINT8   
set8	(pANTLR3_STRING string, const char * chars)
{
    ANTLR3_UINT32	len;

    len = (ANTLR3_UINT32)strlen(chars);
    if	(string->size < len + 1)
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(len + 1));
		string->size	= len + 1;
    }

    /* Note we copy one more byte than the strlen in order to get the trailing '\0'
     */
    ANTLR3_MEMMOVE((void *)(string->chars), chars, (ANTLR3_UINT32)(len+1));
    string->len	    = len;

    return  string->chars;

}

static    pANTLR3_UINT8   
set16_8	(pANTLR3_STRING string, const char * chars)
{
    ANTLR3_UINT32	len;
    ANTLR3_UINT32	count;
    pANTLR3_UINT16	apPoint;

    len = (ANTLR3_UINT32)strlen(chars);
    if	(string->size < len + 1)
	{
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(len + 1)));
		string->size	= len + 1;
    }
    apPoint = ((pANTLR3_UINT16)string->chars);
    string->len	= len;

    for	(count = 0; count < string->len; count++)
    {
		*apPoint++   = *(chars + count);
    }
    *apPoint = '\0';

    return  string->chars;
}

static    pANTLR3_UINT8   
set16_16    (pANTLR3_STRING string, const char * chars)
{
    ANTLR3_UINT32   len;
    pANTLR3_UINT16  in;

    /** First, determine the length of the input string
     */
    in	    = (pANTLR3_UINT16)chars;
    len   = 0;

    while   (*in++ != '\0')
    {
		len++;
    }

    if	(string->size < len + 1)
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(len + 1)));
		string->size	= len + 1;
    }

    /* Note we copy one more byte than the strlen in order to get the trailing '\0'
     */
    ANTLR3_MEMMOVE((void *)(string->chars), chars, (ANTLR3_UINT32)((len+1) * sizeof(ANTLR3_UINT16)));
    string->len	    = len;

    return  string->chars;

}

static    pANTLR3_UINT8   
addc8	(pANTLR3_STRING string, ANTLR3_UINT32 c)
{
    if	(string->size < string->len + 2)
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(string->len + 2));
		string->size	= string->len + 2;
    }
    *(string->chars + string->len)	= (ANTLR3_UINT8)c;
    *(string->chars + string->len + 1)	= '\0';
    string->len++;

    return  string->chars;
}

static    pANTLR3_UINT8   
addc16	(pANTLR3_STRING string, ANTLR3_UINT32 c)
{
    pANTLR3_UINT16  ptr;

    if	(string->size < string->len + 2)
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16) * (string->len + 2)));
		string->size	= string->len + 2;
    }
    ptr	= (pANTLR3_UINT16)(string->chars);

    *(ptr + string->len)	= (ANTLR3_UINT16)c;
    *(ptr + string->len + 1)	= '\0';
    string->len++;

    return  string->chars;
}

static    pANTLR3_UINT8   
addi8	(pANTLR3_STRING string, ANTLR3_INT32 i)
{
    ANTLR3_UINT8	    newbit[32];

    sprintf((char *)newbit, "%d", i);

    return  string->append8(string, (const char *)newbit);
}
static    pANTLR3_UINT8   
addi16	(pANTLR3_STRING string, ANTLR3_INT32 i)
{
    ANTLR3_UINT8	    newbit[32];

    sprintf((char *)newbit, "%d", i);

    return  string->append8(string, (const char *)newbit);
}

static	  pANTLR3_UINT8
inserti8    (pANTLR3_STRING string, ANTLR3_UINT32 point, ANTLR3_INT32 i)
{
    ANTLR3_UINT8	    newbit[32];

    sprintf((char *)newbit, "%d", i);
    return  string->insert8(string, point, (const char *)newbit);
}
static	  pANTLR3_UINT8
inserti16    (pANTLR3_STRING string, ANTLR3_UINT32 point, ANTLR3_INT32 i)
{
    ANTLR3_UINT8	    newbit[32];

    sprintf((char *)newbit, "%d", i);
    return  string->insert8(string, point, (const char *)newbit);
}

static	pANTLR3_UINT8
insert8	(pANTLR3_STRING string, ANTLR3_UINT32 point, const char * newbit)
{
    ANTLR3_UINT32	len;

    if	(point >= string->len)
    {
		return	string->append(string, newbit);
    }
 
    len	= (ANTLR3_UINT32)strlen(newbit);

    if	(len == 0)
    {
		return	string->chars;
    }

    if	(string->size < (string->len + len + 1))
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(string->len + len + 1));
		string->size	= string->len + len + 1;
    }

    /* Move the characters we are inserting before, including the delimiter
     */
    ANTLR3_MEMMOVE((void *)(string->chars + point + len), (void *)(string->chars + point), (ANTLR3_UINT32)(string->len - point + 1));

    /* Note we copy the exact number of bytes
     */
    ANTLR3_MEMMOVE((void *)(string->chars + point), newbit, (ANTLR3_UINT32)(len));
    
    string->len += len;

    return  string->chars;
}

static	pANTLR3_UINT8
insert16_8	(pANTLR3_STRING string, ANTLR3_UINT32 point, const char * newbit)
{
    ANTLR3_UINT32	len;
    ANTLR3_UINT32	count;
    pANTLR3_UINT16	inPoint;

    if	(point >= string->len)
    {
		return	string->append8(string, newbit);
    }
 
    len	= (ANTLR3_UINT32)strlen(newbit);

    if	(len == 0)
    {
		return	string->chars;
    }

    if	(string->size < (string->len + len + 1))
    {
	string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(string->len + len + 1)));
	string->size	= string->len + len + 1;
    }

    /* Move the characters we are inserting before, including the delimiter
     */
    ANTLR3_MEMMOVE((void *)(((pANTLR3_UINT16)string->chars) + point + len), (void *)(((pANTLR3_UINT16)string->chars) + point), (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(string->len - point + 1)));

    string->len += len;
    
    inPoint = ((pANTLR3_UINT16)(string->chars))+point;
    for	(count = 0; count<len; count++)
    {
		*(inPoint + count) = (ANTLR3_UINT16)(*(newbit+count));
    }

    return  string->chars;
}

static	pANTLR3_UINT8
insert16_16	(pANTLR3_STRING string, ANTLR3_UINT32 point, const char * newbit)
{
    ANTLR3_UINT32	len;
    pANTLR3_UINT16	in;

    if	(point >= string->len)
    {
		return	string->append(string, newbit);
    }
 
    /** First, determine the length of the input string
     */
    in	    = (pANTLR3_UINT16)newbit;
    len	    = 0;

    while   (*in++ != '\0')
    {
		len++;
    }

    if	(len == 0)
    {
		return	string->chars;
    }

    if	(string->size < (string->len + len + 1))
    {
		string->chars	= (pANTLR3_UINT8) ANTLR3_REALLOC((void *)string->chars, (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(string->len + len + 1)));
		string->size	= string->len + len + 1;
    }

    /* Move the characters we are inserting before, including the delimiter
     */
    ANTLR3_MEMMOVE((void *)(((pANTLR3_UINT16)string->chars) + point + len), (void *)(((pANTLR3_UINT16)string->chars) + point), (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(string->len - point + 1)));


    /* Note we copy the exact number of characters
     */
    ANTLR3_MEMMOVE((void *)(((pANTLR3_UINT16)string->chars) + point), newbit, (ANTLR3_UINT32)(sizeof(ANTLR3_UINT16)*(len)));
    
    string->len += len;

    return  string->chars;
}

static    pANTLR3_UINT8	    setS	(pANTLR3_STRING string, pANTLR3_STRING chars)
{
    return  string->set(string, (const char *)(chars->chars));
}

static    pANTLR3_UINT8	    appendS	(pANTLR3_STRING string, pANTLR3_STRING newbit)
{
    /* We may be passed an empty string, in which case we just return the current pointer
     */
    if	(newbit == NULL || newbit->len == 0 || newbit->size == 0 || newbit->chars == NULL)
    {
		return	string->chars;
    }
    else
    {
		return  string->append(string, (const char *)(newbit->chars));
    }
}

static	  pANTLR3_UINT8	    insertS	(pANTLR3_STRING string, ANTLR3_UINT32 point, pANTLR3_STRING newbit)
{
    return  string->insert(string, point, (const char *)(newbit->chars));
}

/* Function that compares the text of a string to the supplied
 * 8 bit character string and returns a result a la strcmp()
 */
static ANTLR3_UINT32   
compare8	(pANTLR3_STRING string, const char * compStr)
{
    return  strcmp((const char *)(string->chars), compStr);
}

/* Function that compares the text of a string with the supplied character string
 * (which is assumed to be in the same encoding as the string itself) and returns a result
 * a la strcmp()
 */
static ANTLR3_UINT32   
compare16_8	(pANTLR3_STRING string, const char * compStr)
{
    pANTLR3_UINT16  ourString;
    ANTLR3_UINT32   charDiff;

    ourString	= (pANTLR3_UINT16)(string->chars);

    while   (((ANTLR3_UCHAR)(*ourString) != '\0') && ((ANTLR3_UCHAR)(*compStr) != '\0'))
    {
		charDiff = *ourString - *compStr;
		if  (charDiff != 0)
		{
			return charDiff;
		}
		ourString++;
		compStr++;
    }

    /* At this point, one of the strings was terminated
     */
    return (ANTLR3_UINT32)((ANTLR3_UCHAR)(*ourString) - (ANTLR3_UCHAR)(*compStr));

}

/* Function that compares the text of a string with the supplied character string
 * (which is assumed to be in the same encoding as the string itself) and returns a result
 * a la strcmp()
 */
static ANTLR3_UINT32   
compare16_16	(pANTLR3_STRING string, const char * compStr8)
{
    pANTLR3_UINT16  ourString;
    pANTLR3_UINT16  compStr;
    ANTLR3_UINT32   charDiff;

    ourString	= (pANTLR3_UINT16)(string->chars);
    compStr	= (pANTLR3_UINT16)(compStr8);

    while   (((ANTLR3_UCHAR)(*ourString) != '\0') && ((ANTLR3_UCHAR)(*((pANTLR3_UINT16)compStr)) != '\0'))
    {
		charDiff = *ourString - *compStr;
		if  (charDiff != 0)
		{
			return charDiff;
		}
		ourString++;
		compStr++;
    }

    /* At this point, one of the strings was terminated
     */
    return (ANTLR3_UINT32)((ANTLR3_UCHAR)(*ourString) - (ANTLR3_UCHAR)(*compStr));
}

/* Function that compares the text of a string with the supplied string
 * (which is assumed to be in the same encoding as the string itself) and returns a result
 * a la strcmp()
 */
static ANTLR3_UINT32   
compareS    (pANTLR3_STRING string, pANTLR3_STRING compStr)
{
    return  string->compare(string, (const char *)compStr->chars);
}


/* Function that returns the character indexed at the supplied
 * offset as a 32 bit character.
 */
static ANTLR3_UCHAR    
charAt8	    (pANTLR3_STRING string, ANTLR3_UINT32 offset)
{
    if	(offset > string->len)
    {
		return (ANTLR3_UCHAR)'\0';
    }
    else
    {
		return  (ANTLR3_UCHAR)(*(string->chars + offset));
    }
}

/* Function that returns the character indexed at the supplied
 * offset as a 32 bit character.
 */
static ANTLR3_UCHAR    
charAt16    (pANTLR3_STRING string, ANTLR3_UINT32 offset)
{
    if	(offset > string->len)
    {
		return (ANTLR3_UCHAR)'\0';
    }
    else
    {
		return  (ANTLR3_UCHAR)(*((pANTLR3_UINT16)(string->chars) + offset));
    }
}

/* Function that returns a substring of the supplied string a la .subString(s,e)
 * in java runtimes.
 */
static pANTLR3_STRING
subString8   (pANTLR3_STRING string, ANTLR3_UINT32 startIndex, ANTLR3_UINT32 endIndex)
{
    pANTLR3_STRING newStr;

    if	(endIndex > string->len)
    {
		endIndex = string->len + 1;
    }
    newStr  = string->factory->newPtr(string->factory, string->chars + startIndex, endIndex - startIndex);

    return newStr;
}

/* Returns a substring of the supplied string a la .subString(s,e)
 * in java runtimes.
 */
static pANTLR3_STRING
subString16  (pANTLR3_STRING string, ANTLR3_UINT32 startIndex, ANTLR3_UINT32 endIndex)
{
    pANTLR3_STRING newStr;

    if	(endIndex > string->len)
    {
		endIndex = string->len + 1;
    }
    newStr  = string->factory->newPtr(string->factory, (pANTLR3_UINT8)((pANTLR3_UINT16)(string->chars) + startIndex), endIndex - startIndex);

    return newStr;
}

/* Function that can convert the characters in the string to an integer
 */
static ANTLR3_INT32
toInt32_8	    (struct ANTLR3_STRING_struct * string)
{
    return  atoi((const char *)(string->chars));
}

/* Function that can convert the characters in the string to an integer
 */
static ANTLR3_INT32
toInt32_16       (struct ANTLR3_STRING_struct * string)
{
    pANTLR3_UINT16  input;
    ANTLR3_INT32   value;
    ANTLR3_BOOLEAN  negate;

    value   = 0;
    input   = (pANTLR3_UINT16)(string->chars);
    negate  = ANTLR3_FALSE;

    if	(*input == (ANTLR3_UCHAR)'-')
    {
		negate = ANTLR3_TRUE;
		input++;
    }
    else if (*input == (ANTLR3_UCHAR)'+')
    {
		input++;
    }

    while   (*input != '\0' && isdigit(*input))
    {
		value	 = value * 10;
		value	+= ((ANTLR3_UINT32)(*input) - (ANTLR3_UINT32)'0');
		input++;
    }

    return negate ? -value : value;
}

/* Function that returns a pointer to an 8 bit version of the string,
 * which in this case is just the string as this is 
 * 8 bit encodiing anyway.
 */
static	  pANTLR3_STRING	    to8_8	(pANTLR3_STRING string)
{
    return  string;
}

/* Function that returns an 8 bit version of the string,
 * which in this case is returning all the 16 bit characters
 * narrowed back into 8 bits, with characters that are too large
 * replaced with '_'
 */
static	  pANTLR3_STRING    to8_16	(pANTLR3_STRING string)
{
	pANTLR3_STRING  newStr;
	ANTLR3_UINT32   i;

	/* Create a new 8 bit string
	*/
	newStr  = newRaw8(string->factory);

	if	(newStr == NULL)
	{
		return	NULL;
	}

	/* Always add one more byte for a terminator
	*/
	newStr->chars   = (pANTLR3_UINT8) ANTLR3_MALLOC((size_t)(string->len + 1));
	newStr->size    = string->len + 1;
	newStr->len	    = string->len;

	/* Now copy each 16 bit charActer , making it an 8 bit character of 
	* some sort.
	*/
	for	(i=0; i<string->len; i++)
	{
		ANTLR3_UCHAR	c;

		c = *(((pANTLR3_UINT16)(string->chars)) + i);

		*(newStr->chars + i) = (ANTLR3_UINT8)(c > 255 ? '_' : c);
	}

	/* Terminate
	*/
	*(newStr->chars + newStr->len) = '\0';

	return newStr;
}
