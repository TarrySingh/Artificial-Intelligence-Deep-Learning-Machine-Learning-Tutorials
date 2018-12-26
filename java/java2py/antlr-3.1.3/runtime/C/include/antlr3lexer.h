/** \file 
 * Base interface for any ANTLR3 lexer.
 *
 * An ANLTR3 lexer builds from two sets of components:
 *
 *  - The runtime components that provide common functionality such as
 *    traversing character streams, building tokens for output and so on.
 *  - The generated rules and struutre of the actual lexer, which call upon the
 *    runtime components.
 *
 * A lexer class contains  a character input stream, a base recognizer interface 
 * (which it will normally implement) and a token source interface (which it also
 * implements. The Tokensource interface is called by a token consumer (such as
 * a parser, but in theory it can be anything that wants a set of abstract
 * tokens in place of a raw character stream.
 *
 * So then, we set up a lexer in a sequence akin to:
 *
 *  - Create a character stream (something which implements ANTLR3_INPUT_STREAM)
 *    and initialize it.
 *  - Create a lexer interface and tell it where it its input stream is.
 *    This will cause the creation of a base recognizer class, which it will 
 *    override with its own implementations of some methods. The lexer creator
 *    can also then in turn override anything it likes. 
 *  - The lexer token source interface is then passed to some interface that
 *    knows how to use it, byte calling for a next token. 
 *  - When a next token is called, let ze lexing begin.
 *
 */
#ifndef	_ANTLR3_LEXER
#define	_ANTLR3_LEXER

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

/* Definitions
 */
#define	ANTLR3_STRING_TERMINATOR	0xFFFFFFFF

#include    <antlr3defs.h>
#include    <antlr3input.h>
#include    <antlr3commontoken.h>
#include    <antlr3tokenstream.h>
#include    <antlr3baserecognizer.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef	struct ANTLR3_LEXER_struct
{
    /** If there is a super structure that is implementing the
     *  lexer, then a pointer to it can be stored here in case
     *  implementing functions are overridden by this super structure.
     */
    void	* super;

    /** A generated lexer has an mTokens() function, which needs
     *  the context pointer of the generated lexer, not the base lexer interface
     *  this is stored here and initialized by the generated code (or manually
     *  if this is a manually built lexer.
     */
    void	* ctx;

    /** A pointer to the character stream whence this lexer is receiving
     *  characters. 
     *  TODO: I may come back to this and implement charstream outside
     *  the input stream as per the java implementation.
     */
    pANTLR3_INPUT_STREAM	input;

    /** Pointer to the implementation of a base recognizer, which the lexer
     *  creates and then overrides with its own lexer oriented functions (the 
     *  default implementation is parser oriented). This also contains a
     *  token source interface, which the lexer instance will provide to anything 
     *  that needs it, which is anything else that implements a base recognizer,
     *  such as a parser.
     */
    pANTLR3_BASE_RECOGNIZER	rec;

    /** Pointer to a function that sets the charstream source for the lexer and
     *  causes it to  be reset.
     */
    void			(*setCharStream)    (struct ANTLR3_LEXER_struct * lexer, pANTLR3_INPUT_STREAM input);
    
    /** Pointer to a function that switches the current character input stream to 
     *  a new one, saving the old one, which we will revert to at the end of this 
     *  new one.
     */
    void			(*pushCharStream)   (struct ANTLR3_LEXER_struct * lexer, pANTLR3_INPUT_STREAM input);

    /** Pointer to a function that abandons the current input stream, whether it
     *  is empty or not and reverts to the previous stacked input stream.
     */
    void			(*popCharStream)    (struct ANTLR3_LEXER_struct * lexer);

    /** Pointer to a function that emits the supplied token as the next token in
     *  the stream.
     */
    void			(*emitNew)	    (struct ANTLR3_LEXER_struct * lexer, pANTLR3_COMMON_TOKEN token);

    /** Pointer to a function that constructs a new token from the lexer stored information 
     */
    pANTLR3_COMMON_TOKEN	(*emit)		    (struct ANTLR3_LEXER_struct * lexer);

    /** Pointer to the user provided (either manually or through code generation
     *  function that causes the lexer rules to run the lexing rules and produce 
     *  the next token if there iss one. This is called from nextToken() in the
     *  pANTLR3_TOKEN_SOURCE. Note that the input parameter for this funciton is 
     *  the generated lexer context (stored in ctx in this interface) it is a generated
     *  function and expects the context to be the generated lexer. 
     */
    void	        (*mTokens)		    (void * ctx);

    /** Pointer to a function that attempts to match and consume the specified string from the input
     *  stream. Note that strings muse be passed as terminated arrays of ANTLR3_UCHAR. Strings are terminated
     *  with 0xFFFFFFFF, which is an invalid UTF32 character
     */
    ANTLR3_BOOLEAN	(*matchs)	    (struct ANTLR3_LEXER_struct * lexer, ANTLR3_UCHAR * string);

    /** Pointer to a function that matches and consumes the specified character from the input stream.
     *  As the input stream is required to provide characters via LA() as UTF32 characters it does not 
     *  need to provide an implementation if it is not sourced from 8 bit ASCII. The default lexer
     *  implementation is source encoding agnostic, unless for some reason it takes two 32 bit characters
     *  to specify a single character, in which case the input stream and the lexer rules would have to match
     *  in encoding and then it would work 'by accident' anyway.
     */
    ANTLR3_BOOLEAN	(*matchc)	    (struct ANTLR3_LEXER_struct * lexer, ANTLR3_UCHAR c);

    /** Pointer to a function that matches any character in the supplied range (I suppose it could be a token range too
     *  but this would only be useful if the tokens were in tsome guaranteed order which is
     *  only going to happen with a hand crafted token set).
     */
    ANTLR3_BOOLEAN	(*matchRange)	    (struct ANTLR3_LEXER_struct * lexer, ANTLR3_UCHAR low, ANTLR3_UCHAR high);

    /** Pointer to a function that matches the next token/char in the input stream
     *  regardless of what it actaully is.
     */
    void		(*matchAny)	    (struct ANTLR3_LEXER_struct * lexer);

    /** Pointer to a function that recovers from an error found in the input stream.
     *  Generally, this will be a #ANTLR3_EXCEPTION_NOVIABLE_ALT but it could also
     *  be from a mismatched token that the (*match)() could not recover from.
     */
    void		(*recover)	    (struct ANTLR3_LEXER_struct * lexer);

    /** Pointer to function to return the current line number in the input stream
     */
    ANTLR3_UINT32	(*getLine)		(struct ANTLR3_LEXER_struct * lexer);
    ANTLR3_MARKER	(*getCharIndex)		(struct ANTLR3_LEXER_struct * lexer);
    ANTLR3_UINT32	(*getCharPositionInLine)(struct ANTLR3_LEXER_struct * lexer);

    /** Pointer to function to return the text so far for the current token being generated
     */
    pANTLR3_STRING	(*getText)	    (struct ANTLR3_LEXER_struct * lexer);


    /** Pointer to a function that knows how to free the resources of a lexer
     */
    void		(*free)		    (struct ANTLR3_LEXER_struct * lexer);

}
    ANTLR3_LEXER;

#ifdef __cplusplus
}
#endif

#endif
