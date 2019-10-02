/// \file
/// Base functions to initialize and manipulate a UCS2 input stream
///
#include    <antlr3input.h>

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

// INT Stream API
//
static	    void			antlr3UCS2Consume	(pANTLR3_INT_STREAM is);
static	    ANTLR3_UCHAR    antlr3UCS2LA		(pANTLR3_INT_STREAM is, ANTLR3_INT32 la);
static	    ANTLR3_MARKER   antlr3UCS2Index		(pANTLR3_INT_STREAM is);
static	    void			antlr3UCS2Seek		(pANTLR3_INT_STREAM is, ANTLR3_MARKER seekPoint);

// ucs2 Charstream API functions
//
static	    pANTLR3_STRING	antlr3UCS2Substr	(pANTLR3_INPUT_STREAM input, ANTLR3_MARKER start, ANTLR3_MARKER stop);

/// \brief Common function to setup function interface for a 16 bit "UCS2" input stream.
///
/// \param input Input stream context pointer
///
/// \remark
///  - Strictly speaking, there is no such thing as a UCS2 input stream as the term
///    tends to confuse the notions of character encoding, unicode and so on. However
///    because there will possibly be a need for a UTF-16 stream, I needed to identify 16 bit
///    streams that do not support surrogate encodings and UCS2 is how it is mostly referred to.
///    For instance Java, Oracle and others use a 16 bit encoding of characters and so this type
///    of stream is very common.
///    Take it to mean, therefore, a straight 16 bit uncomplicated encoding of Unicode code points.
///
void 
antlr3UCS2SetupStream	(pANTLR3_INPUT_STREAM input, ANTLR3_UINT32 type)
{
    // Build a string factory for this stream. This is a 16 bit string "UCS2" factory which is a standard
    // part of the ANTLR3 string. The string factory is then passed through the whole chain of lexer->parser->tree->treeparser
    // and so on.
    //
    input->strFactory	= antlr3UCS2StringFactoryNew();

    // Install function pointers for an 8 bit ASCII input, which are good for almost
    // all input stream functions. We will then override those that won't work for 16 bit characters.
    //
    antlr3GenericSetupStream	(input, type);

    // Intstream API overrides for UCS2
    //
    input->istream->consume	    =  antlr3UCS2Consume;	    // Consume the next 16 bit character in the buffer
    input->istream->_LA		    =  antlr3UCS2LA;		    // Return the UTF32 character at offset n (1 based)
    input->istream->index	    =  antlr3UCS2Index;		    // Calculate current index in input stream, 16 bit based
    input->istream->seek	    =  antlr3UCS2Seek;		    // How to seek to a specific point in the stream
    
    // Charstream API overrides for UCS2
    //
    input->substr		    =  antlr3UCS2Substr;	    // Return a string from the input stream
        
	input->charByteSize				= 2;				// Size in bytes of characters in this stream.

}

/// \brief Consume the next character in an 8 bit ASCII input stream
///
/// \param input Input stream context pointer
///
static void
antlr3UCS2Consume(pANTLR3_INT_STREAM is)
{
	pANTLR3_INPUT_STREAM input;

	input   = ((pANTLR3_INPUT_STREAM) (is->super));

	if	((pANTLR3_UINT16)(input->nextChar) < (((pANTLR3_UINT16)input->data) + input->sizeBuf))
	{	
		// Indicate one more character in this line
		//
		input->charPositionInLine++;

		if  ((ANTLR3_UCHAR)(*((pANTLR3_UINT16)input->nextChar)) == input->newlineChar)
		{
			// Reset for start of a new line of input
			//
			input->line++;
			input->charPositionInLine	= 0;
			input->currentLine		= (void *)(((pANTLR3_UINT16)input->nextChar) + 1);
		}

		// Increment to next character position
		//
		input->nextChar = (void *)(((pANTLR3_UINT16)input->nextChar) + 1);
	}
}

/// \brief Return the input element assuming an 8 bit ascii input
///
/// \param[in] input Input stream context pointer
/// \param[in] la 1 based offset of next input stream element
///
/// \return Next input character in internal ANTLR3 encoding (UTF32)
///
static ANTLR3_UCHAR 
antlr3UCS2LA(pANTLR3_INT_STREAM is, ANTLR3_INT32 la)
{
	pANTLR3_INPUT_STREAM input;

	input   = ((pANTLR3_INPUT_STREAM) (is->super));

	if	(( ((pANTLR3_UINT16)input->nextChar) + la - 1) >= (((pANTLR3_UINT16)input->data) + input->sizeBuf))
	{
		return	ANTLR3_CHARSTREAM_EOF;
	}
	else
	{
		return	(ANTLR3_UCHAR)(*((pANTLR3_UINT16)input->nextChar + la - 1));
	}
}


/// \brief Calculate the current index in the output stream.
/// \param[in] input Input stream context pointer
///
static ANTLR3_MARKER 
antlr3UCS2Index(pANTLR3_INT_STREAM is)
{
    pANTLR3_INPUT_STREAM input;

    input   = ((pANTLR3_INPUT_STREAM) (is->super));

    return  (ANTLR3_MARKER)(input->nextChar);
}

/// \brief Rewind the lexer input to the state specified by the supplied mark.
///
/// \param[in] input Input stream context pointer
///
/// \remark
/// Assumes ASCII (or at least, 8 Bit) input stream.
///
static void
antlr3UCS2Seek	(pANTLR3_INT_STREAM is, ANTLR3_MARKER seekPoint)
{
	ANTLR3_INT32   count;
	pANTLR3_INPUT_STREAM input;

	input   = ((pANTLR3_INPUT_STREAM) is->super);

	// If the requested seek point is less than the current
	// input point, then we assume that we are resetting from a mark
	// and do not need to scan, but can just set to there.
	//
	if	(seekPoint <= (ANTLR3_MARKER)(input->nextChar))
	{
		input->nextChar	= (void *)seekPoint;
	}
	else
	{
		count	= (ANTLR3_UINT32)((seekPoint - (ANTLR3_MARKER)(input->nextChar)) / 2); // 16 bits per character in UCS2

		while (count--)
		{
			is->consume(is);
		}
	}
}
/// \brief Return a substring of the ucs2 (16 bit) input stream in
///  newly allocated memory.
///
/// \param input Input stream context pointer
/// \param start Offset in input stream where the string starts
/// \param stop  Offset in the input stream where the string ends.
///
static pANTLR3_STRING
antlr3UCS2Substr		(pANTLR3_INPUT_STREAM input, ANTLR3_MARKER start, ANTLR3_MARKER stop)
{
    return  input->strFactory->newPtr(input->strFactory, (pANTLR3_UINT8)start, ((ANTLR3_UINT32_CAST(stop - start))/2) + 1);
}
