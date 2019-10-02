/** \file
 *  Contains the definition of a basic ANTLR3 exception structure created
 *  by a recognizer when errors are found/predicted.
 */
#ifndef	_ANTLR3_EXCEPTION_H
#define	_ANTLR3_EXCEPTION_H

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

/** Indicates that the recognizer received a token
 *  in the input that was not predicted.
 */
#define	ANTLR3_RECOGNITION_EXCEPTION	    1

/** Name of exception #ANTLR3_RECOGNITION_EXCEPTION
 */
#define	ANTLR3_RECOGNITION_EX_NAME  "Recognition Exception"

/** Indicates that the recognizer was expecting one token and found a
 *  a different one.
 */
#define	ANTLR3_MISMATCHED_TOKEN_EXCEPTION   2

/** Name of #ANTLR3_MISMATCHED_TOKEN_EXCEPTION
 */
#define	ANTLR3_MISMATCHED_EX_NAME   "Mismatched Token Exception"

/** Recognizer could not find a valid alternative from the input
 */
#define	ANTLR3_NO_VIABLE_ALT_EXCEPTION	    3

/** Name of #ANTLR3_NO_VIABLE_ALT_EXCEPTION
 */
#define	ANTLR3_NO_VIABLE_ALT_NAME   "No Viable Alt"

/* Character in a set was not found
 */
#define	ANTLR3_MISMATCHED_SET_EXCEPTION	    4

/* Name of #ANTLR3_MISMATCHED_SET_EXCEPTION
 */
#define	ANTLR3_MISMATCHED_SET_NAME  "Mismatched set"

/* A rule predicting at least n elements found less than that,
 * such as: WS: " "+;
 */
#define	ANTLR3_EARLY_EXIT_EXCEPTION	    5

/* Name of #ANTLR3_EARLY_EXIT_EXCEPTION
 */
#define	ANTLR3_EARLY_EXIT_NAME	     "Early exit"

#define	ANTLR3_FAILED_PREDICATE_EXCEPTION   6
#define	ANTLR3_FAILED_PREDICATE_NAME	    "Predicate failed!"

#define	ANTLR3_MISMATCHED_TREE_NODE_EXCEPTION	7
#define	ANTLR3_MISMATCHED_TREE_NODE_NAME    "Mismatched tree node!"

#define	ANTLR3_REWRITE_EARLY_EXCEPTION	8
#define	ANTLR3_REWRITE_EARLY_EXCEPTION_NAME    "Mismatched tree node!"

#define	ANTLR3_UNWANTED_TOKEN_EXCEPTION	9
#define	ANTLR3_UNWANTED_TOKEN_EXCEPTION_NAME    "Extraneous token"

#define	ANTLR3_MISSING_TOKEN_EXCEPTION	10
#define	ANTLR3_MISSING_TOKEN_EXCEPTION_NAME    "Missing token"

#ifdef __cplusplus
extern "C" {
#endif

/** Base structure for an ANTLR3 exception tracker
 */
typedef	struct ANTLR3_EXCEPTION_struct
{
	/// Set to one of the exception type defines:
	///
	///  - #ANTLR3_RECOGNITION_EXCEPTION
	///  - #ANTLR3_MISMATCHED_TOKEN_EXCEPTION
	///  - #ANTLR3_NO_VIABLE_ALT_EXCEPTION
	///  - #ANTLR3_MISMATCHED_SET_EXCEPTION
	///  - #ANTLR3_EARLY_EXIT_EXCEPTION
	///  - #ANTLR3_FAILED_PREDICATE_EXCEPTION
	///  - #ANTLR3_EARLY_EXIT_EXCEPTION
    ///
    ANTLR3_UINT32   type;

    /** The string name of the exception
     */
    void    *	    name;

    /** The printable message that goes with this exception, in your preferred
     *  encoding format. ANTLR just uses ASCII by default but you can ignore these
     *  messages or convert them to another format or whatever of course. They are
     *  really internal messages that you then decide how to print out in a form that
     *  the users of your product will understand, as they are unlikely to know what
     *  to do with "Recognition exception at: [[TOK_GERUND..... " ;-)
     */
    void    *	    message;

    /** Name of the file/input source for reporting. Note that this may be NULL!!
     */
    pANTLR3_STRING streamName;

    /** If set to ANTLR3_TRUE, this indicates that the message element of this structure
     *  should be freed by calling ANTLR3_FREE() when the exception is destroyed.
     */
    ANTLR3_BOOLEAN  freeMessage;

    /** Indicates the index of the 'token' we were looking at when the
     *  exception occurred.
     */
    ANTLR3_MARKER  index;

    /** Indicates what the current token/tree was when the error occurred. Since not
     *  all input streams will be able to retrieve the nth token, we track it here
     *  instead. This is for parsers, and even tree parsers may set this.
     */
    void	* token;

    /** Indicates the token we were expecting to see next when the error occurred
     */
    ANTLR3_UINT32   expecting;

    /** Indicates a set of tokens that we were expecting to see one of when the
     *  error occurred. It is a following bitset list, so you can use load it and use ->toIntList() on it
     *  to generate an array of integer tokens that it represents.
     */
    pANTLR3_BITSET_LIST  expectingSet;

    /** If this is a tree parser exception then the node is set to point to the node
     * that caused the issue.
     */
    void	* node;

    /** The current character when an error occurred - for lexers.
     */
    ANTLR3_UCHAR   c;

    /** Track the line at which the error occurred in case this is
     *  generated from a lexer.  We need to track this since the
     *  unexpected char doesn't carry the line info.
     */
    ANTLR3_UINT32   line;

    /** Character position in the line where the error occurred.
     */
    ANTLR3_INT32   charPositionInLine;

    /** decision number for NVE
     */
    ANTLR3_UINT32   decisionNum;

    /** State for NVE
     */
    ANTLR3_UINT32   state;

    /** Rule name for failed predicate exception
     */
    void	    * ruleName;

    /** Pointer to the next exception in the chain (if any)
     */
    struct ANTLR3_EXCEPTION_struct * nextException;

    /** Pointer to the input stream that this exception occurred in.
     */
    pANTLR3_INT_STREAM    input;

    /** Pointer for you, the programmer to add anything you like to an exception.
     */
    void    *	    custom;

    /** Pointer to a routine that is called to free the custom exception structure
     *  when the exception is destroyed. Set to NULL if nothing should be done.
     */
    void	    (*freeCustom)   (void * custom);
    void	    (*print)	    (struct ANTLR3_EXCEPTION_struct * ex);
    void	    (*freeEx)	    (struct ANTLR3_EXCEPTION_struct * ex);

}
    ANTLR3_EXCEPTION;

#ifdef __cplusplus
}
#endif


#endif
