/** \file
 * Defines the basic structures used to manipulate character
 * streams from any input source. The first implementation of
 * this stream was ASCII 8 bit, but any character size and encoding
 * can in theory be used, so long as they can return a 32 bit Integer
 * representation of their characters amd efficiently mark and revert
 * to specific offsets into their input streams.
 */
#ifndef	_ANTLR3_INPUT_H
#define	_ANTLR3_INPUT_H

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

#include    <antlr3defs.h>
#include    <antlr3string.h>
#include    <antlr3commontoken.h>
#include    <antlr3intstream.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Master context structure for an ANTLR3 C runtime based input stream.
/// \ingroup apistructures
///
typedef	struct	ANTLR3_INPUT_STREAM_struct
{
    /** Interfaces that provide streams must all provide
     *  a generic ANTLR3_INT_STREAM interface and an ANTLR3_INPUT_STREAM
     *  is no different.
     */
    pANTLR3_INT_STREAM	istream;

    /** Whatever super structure is providing the INPUT stream needs a pointer to itself
     *  so that this can be passed back to it whenever the api functions
     *  are called back from this interface.
     */
    void	      * super;

	/// Indicates the size, in 8 bit units, of a single character. Note that
	/// the C runtime does not deal with surrogates and UTF8 directly as this would be
	/// slow and complicated. Variable character width inputs are expected to be converted
	/// into fixed width formats, so that would be a UTF32 format for anything that cannot
	/// work with a UCS2 encoding, such as UTF-8. Generally you are best
	/// working internally with 32 bit characters.
	///
	ANTLR3_UINT8	charByteSize;

    /** Pointer the start of the input string, characters may be
     *  taken as offsets from here and in original input format encoding.
     */
    void	      *	data;

    /** Indicates if the data pointer was allocated by us, and so should be freed
     *  when the stream dies.
     */
    int			isAllocated;

    /** String factory for this input stream
     */
    pANTLR3_STRING_FACTORY  strFactory;


    /** Pointer to the next character to be consumed from the input data
     *  This is cast to point at the encoding of the original file that
     *  was read by the functions installed as pointer in this input stream
     *  context instance at file/string/whatever load time.
     */
    void	      * nextChar;

    /** Number of characters that can be consumed at this point in time.
     *  Mostly this is just what is left in the pre-read buffer, but if the
     *  input source is a stream such as a socket or something then we may
     *  call special read code to wait for more input.
     */
    ANTLR3_UINT32	sizeBuf;

    /** The line number we are traversing in the input file. This gets incremented
     *  by a newline() call in the lexer grammar actions.
     */
    ANTLR3_UINT32	line;

    /** Pointer into the input buffer where the current line
     *  started.
     */
    void	      * currentLine;

    /** The offset within the current line of the current character
     */
    ANTLR3_INT32	charPositionInLine;

    /** Tracks how deep mark() calls are nested
     */
    ANTLR3_UINT32	markDepth;

    /** List of mark() points in the input stream
     */
    pANTLR3_VECTOR	markers;

    /** File name string, set to pointer to memory if
     * you set it manually as it will be free()d
     */
    pANTLR3_STRING	fileName;

	/** File number, needs to be set manually to some file index of your devising.
	 */
	ANTLR3_UINT32	fileNo;

    /** Character that automatically causes an internal line count
     *  increment.
     */
    ANTLR3_UCHAR	newlineChar;

    /* API */


   /** Pointer to function that closes the input stream
     */
    void		(*close)	(struct	ANTLR3_INPUT_STREAM_struct * input);
    void		(*free)		(struct	ANTLR3_INPUT_STREAM_struct * input);

    /** Pointer to function that resets the input stream
     */
    void		(*reset)	(struct	ANTLR3_INPUT_STREAM_struct * input);

	/**
	 * Pinter to function that installs a version of LA that always
	 * returns upper case. Only valid for character streams and creates a case
	 * insensitive lexer if the lexer tokens are described in upper case. The
	 * tokens will preserve case in the token text.
	 */
	void		(*setUcaseLA)		(pANTLR3_INPUT_STREAM input, ANTLR3_BOOLEAN flag);

    /** Pointer to function to return input stream element at 1 based
     *  offset from nextChar. Same as _LA for char stream, but token
     *  streams etc. have one of these that does other stuff of course.
     */
    void *		(*_LT)		(struct	ANTLR3_INPUT_STREAM_struct * input, ANTLR3_INT32 lt);

    /** Pointer to function to return the total size of the input buffer. For streams
     *  this may be just the total we have available so far. This means of course that
     *  the input stream must be careful to accumulate enough input so that any backtracking
     *  can be satisfied.
     */
    ANTLR3_UINT32	(*size)		(struct ANTLR3_INPUT_STREAM_struct * input);

    /** Pointer to function to return a substring of the input stream. String is returned in allocated
     *  memory and is in same encoding as the input stream itself, NOT internal ANTLR3_UCHAR form.
     */
    pANTLR3_STRING	(*substr)	(struct ANTLR3_INPUT_STREAM_struct * input, ANTLR3_MARKER start, ANTLR3_MARKER stop);

    /** Pointer to function to return the current line number in the input stream
     */
    ANTLR3_UINT32	(*getLine)	(struct ANTLR3_INPUT_STREAM_struct * input);

    /** Pointer to function to return the current line buffer in the input stream
     *  The pointer returned is directly into the input stream so you must copy
     *  it if you wish to manipulate it without damaging the input stream. Encoding
     *  is obviously in the same form as the input stream.
     *  \remark
     *    - Note taht this function wil lbe inaccurate if setLine is called as there
     *      is no way at the moment to position the input stream at a particular line 
     *	    number offset.
     */
    void	  *	(*getLineBuf)	(struct ANTLR3_INPUT_STREAM_struct * input);

    /** Pointer to function to return the current offset in the current input stream line
     */
    ANTLR3_UINT32	(*getCharPositionInLine)  (struct ANTLR3_INPUT_STREAM_struct * input);

    /** Pointer to function to set the current line number in the input stream
     */
    void		(*setLine)		  (struct ANTLR3_INPUT_STREAM_struct * input, ANTLR3_UINT32 line);

    /** Pointer to function to set the current position in the current line.
     */
    void		(*setCharPositionInLine)  (struct ANTLR3_INPUT_STREAM_struct * input, ANTLR3_UINT32 position);

    /** Pointer to function to override the default newline character that the input stream
     *  looks for to trigger the line and offset and line buffer recording information.
     *  \remark
     *   - By default the chracter '\n' will be instaleldas tehe newline trigger character. When this
     *     character is seen by the consume() function then the current line number is incremented and the
     *     current line offset is reset to 0. The Pointer for the line of input we are consuming
     *     is updated to point to the next character after this one in the input stream (which means it
     *     may become invlaid if the last newline character in the file is seen (so watch out).
     *   - If for some reason you do not want teh counters and pointesr to be restee, yu can set the 
     *     chracter to some impossible charater such as '\0' or whatever.
     *   - This is a single character only, so choose the last chracter in a sequence of two or more.
     *   - This is only a simple aid to error reporting - if you have a complicated binary inptu structure
     *     it may not be adequate, but you can always override every function in the input stream with your
     *     own of course, and can even write your own complete input stream set if you like.
     *   - It is your responsiblity to set a valid cahracter for the input stream type. Ther is no point 
     *     setting this to 0xFFFFFFFF if the input stream is 8 bit ASCII as this will just be truncated and never
     *	   trigger as the comparison will be (INT32)0xFF == (INT32)0xFFFFFFFF
     */
    void		(*SetNewLineChar)	    (struct ANTLR3_INPUT_STREAM_struct * input, ANTLR3_UINT32 newlineChar);

}

    ANTLR3_INPUT_STREAM;


/** \brief Structure for track lex input states as part of mark()
 *  and rewind() of lexer.
 */
typedef	struct	ANTLR3_LEX_STATE_struct
{
        /** Pointer to the next character to be consumed from the input data
     *  This is cast to point at the encoding of the original file that
     *  was read by the functions installed as pointer in this input stream
     *  context instance at file/string/whatever load time.
     */
    void	      * nextChar;

    /** The line number we are traversing in the input file. This gets incremented
     *  by a newline() call in the lexer grammer actions.
     */
    ANTLR3_UINT32	line;

    /** Pointer into the input buffer where the current line
     *  started.
     */
    void	      * currentLine;

    /** The offset within the current line of the current character
     */
    ANTLR3_INT32	charPositionInLine;

}
    ANTLR3_LEX_STATE;

    /* Prototypes 
     */
    void	    antlr3AsciiSetupStream	(pANTLR3_INPUT_STREAM input, ANTLR3_UINT32 type);
    void	    antlr3UCS2SetupStream	(pANTLR3_INPUT_STREAM input, ANTLR3_UINT32 type);
    void	    antlr3GenericSetupStream	(pANTLR3_INPUT_STREAM input, ANTLR3_UINT32 type);

#ifdef __cplusplus
}
#endif

#endif	/* _ANTLR3_INPUT_H  */
