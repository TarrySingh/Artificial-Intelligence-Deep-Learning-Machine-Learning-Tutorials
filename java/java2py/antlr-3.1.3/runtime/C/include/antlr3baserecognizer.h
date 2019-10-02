/** \file
 * Defines the basic structure to support recognizing by either a lexer,
 * parser, or tree parser.
 * \addtogroup ANTLR3_BASE_RECOGNIZER
 * @{
 */
#ifndef	_ANTLR3_BASERECOGNIZER_H
#define	_ANTLR3_BASERECOGNIZER_H

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
#include    <antlr3exception.h>
#include    <antlr3input.h>
#include    <antlr3tokenstream.h>
#include    <antlr3commontoken.h>
#include    <antlr3commontreenodestream.h>
#include	<antlr3debugeventlistener.h>
#include	<antlr3recognizersharedstate.h>

/** Type indicator for a lexer recognizer
 */
#define	    ANTLR3_TYPE_LEXER		0x0001

/** Type indicator for a parser recognizer
 */
#define	    ANTLR3_TYPE_PARSER		0x0002

/** Type indicator for a tree parser recognizer
 */
#define	    ANTLR3_TYPE_TREE_PARSER	0x0004

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Base tracking context structure for all types of
 * recognizers.
 */
typedef	struct ANTLR3_BASE_RECOGNIZER_struct
{
    /// Whatever super structure is providing this interface needs a pointer to itself
    /// so that this can be passed back to it whenever the api functions
    /// are called back from here.
    ///
    void	      * super;
    
	/// Indicates the type of recognizer that we are an instance of.
    /// The programmer may set this to anything of course, but the default 
    /// implementations of the interface only really understand the built in
    /// types, so new error handlers etc would probably be required to as well.
    /// 
    ///  Valid types are:
    ///
    ///   - #ANTLR3_TYPE_LEXER  
	///	  - #ANTLR3_TYPE_PARSER
    ///   - #ANTLR3_TYPE_TREE_PARSER
    ///
    ANTLR3_UINT32	type;

	/// A pointer to the shared recognizer state, such that multiple
	/// recognizers can use the same inputs streams and so on (in
	/// the case of grammar inheritance for instance.
	///
	pANTLR3_RECOGNIZER_SHARED_STATE	state;

	/// If set to something other than NULL, then this structure is
	/// points to an instance of the debugger interface. In general, the
	/// debugger is only referenced internally in recovery/error operations
	/// so that it does not cause overhead by having to check this pointer
	/// in every function/method
	///
	pANTLR3_DEBUG_EVENT_LISTENER	debugger;


    /// Pointer to a function that matches the current input symbol
    /// against the supplied type. the function causes an error if a
    /// match is not found and the default implementation will also
    /// attempt to perform one token insertion or deletion if that is
    /// possible with the input stream. You can override the default
    /// implementation by installing a pointer to your own function
    /// in this interface after the recognizer has initialized. This can
    /// perform different recovery options or not recover at all and so on.
    /// To ignore recovery altogether, see the comments in the default
    /// implementation of this function in antlr3baserecognizer.c
    ///
    /// Note that errors are signalled by setting the error flag below
    /// and creating a new exception structure and installing it in the
    /// exception pointer below (you can chain these if you like and handle them
    /// in some customized way).
    ///
    void *		(*match)	(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
							    ANTLR3_UINT32 ttype, pANTLR3_BITSET_LIST follow);

    /// Pointer to a function that matches the next token/char in the input stream
    /// regardless of what it actually is.
    ///
    void		(*matchAny)	(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);
    
	/// Pointer to a function that decides if the token ahead of the current one is the 
	/// one we were loking for, in which case the curernt one is very likely extraneous
	/// and can be reported that way.
	///
	ANTLR3_BOOLEAN
				(*mismatchIsUnwantedToken)	(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer, pANTLR3_INT_STREAM input, ANTLR3_UINT32 ttype);

	/// Pointer to a function that decides if the current token is one that can logically
	/// follow the one we were looking for, in which case the one we were looking for is 
	/// probably missing from the input.
	///
	ANTLR3_BOOLEAN
				(*mismatchIsMissingToken)	(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer, pANTLR3_INT_STREAM input, pANTLR3_BITSET_LIST follow);

    /** Pointer to a function that works out what to do when a token mismatch
     *  occurs, so that Tree parsers can behave differently to other recognizers.
     */
    void		(*mismatch)	(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
					    ANTLR3_UINT32 ttype, pANTLR3_BITSET_LIST follow);

    /** Pointer to a function to call to report a recognition problem. You may override
     *  this function with your own function, but refer to the standard implementation
     *  in antlr3baserecognizer.c for guidance. The function should recognize whether 
     *  error recovery is in force, so that it does not print out more than one error messages
     *  for the same error. From the java comments in BaseRecognizer.java:
     *
     *  This method sets errorRecovery to indicate the parser is recovering
     *  not parsing.  Once in recovery mode, no errors are generated.
     *  To get out of recovery mode, the parser must successfully match
     *  a token (after a resync).  So it will go:
     *
     * 		1. error occurs
     * 		2. enter recovery mode, report error
     * 		3. consume until token found in resynch set
     * 		4. try to resume parsing
     * 		5. next match() will reset errorRecovery mode
     */
    void		(*reportError)		    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Pointer to a function that is called to display a recognition error message. You may
     *  override this function independently of (*reportError)() above as that function calls
     *  this one to do the actual exception printing.
     */
    void		(*displayRecognitionError)  (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer, pANTLR3_UINT8 * tokenNames);

	/// Get number of recognition errors (lexer, parser, tree parser).  Each
	/// recognizer tracks its own number.  So parser and lexer each have
	/// separate count.  Does not count the spurious errors found between
	/// an error and next valid token match
	///
	/// \see reportError()
	///
	ANTLR3_UINT32
				(*getNumberOfSyntaxErrors)	(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Pointer to a function that recovers from an error found in the input stream.
     *  Generally, this will be a #ANTLR3_EXCEPTION_NOVIABLE_ALT but it could also
     *  be from a mismatched token that the (*match)() could not recover from.
     */
    void		(*recover)		    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Pointer to a function that is a hook to listen to token consumption during error recovery.
     *  This is mainly used by the debug parser to send events to the listener.
     */
    void		(*beginResync)		    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Pointer to a function that is a hook to listen to token consumption during error recovery.
     *  This is mainly used by the debug parser to send events to the listener.
     */
    void		(*endResync)		    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

	/** Pointer to a function that is a hook to listen to token consumption during error recovery.
     *  This is mainly used by the debug parser to send events to the listener.
     */
    void		(*beginBacktrack)		(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer, ANTLR3_UINT32 level);

    /** Pointer to a function that is a hook to listen to token consumption during error recovery.
     *  This is mainly used by the debug parser to send events to the listener.
     */
    void		(*endBacktrack)		    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer, ANTLR3_UINT32 level, ANTLR3_BOOLEAN successful);

    /** Pointer to a function to computer the error recovery set for the current rule.
     *  \see antlr3ComputeErrorRecoverySet() for details.
     */
    pANTLR3_BITSET	(*computeErrorRecoverySet)  (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Pointer to a function that computes the context-sensitive FOLLOW set for the 
     *  current rule.
     * \see antlr3ComputeCSRuleFollow() for details.
     */
    pANTLR3_BITSET	(*computeCSRuleFollow)	    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Pointer to a function to combine follow bitsets.
     * \see antlr3CombineFollows() for details.
     */
    pANTLR3_BITSET	(*combineFollows)	    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer, 
							    ANTLR3_BOOLEAN exact);
 
    /** Pointer to a function that recovers from a mismatched token in the input stream.
     * \see antlr3RecoverMismatch() for details.
     */
    void		* (*recoverFromMismatchedToken)
						    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
							    ANTLR3_UINT32	ttype,
							    pANTLR3_BITSET_LIST	follow);

    /** Pointer to a function that recovers from a mismatched set in the token stream, in a similar manner
     *  to (*recoverFromMismatchedToken)
     */
    void		* (*recoverFromMismatchedSet) (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
							    pANTLR3_BITSET_LIST	follow);

    /** Pointer to common routine to handle single token insertion for recovery functions.
     */
    ANTLR3_BOOLEAN	(*recoverFromMismatchedElement)
						    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
							    pANTLR3_BITSET_LIST	follow);
    
    /** Pointer to function that consumes input until the next token matches
     *  the given token.
     */
    void		(*consumeUntil)		    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
							    ANTLR3_UINT32   tokenType);

    /** Pointer to function that consumes input until the next token matches
     *  one in the given set.
     */
    void		(*consumeUntilSet)	    (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
							    pANTLR3_BITSET	set);

    /** Pointer to function that returns an ANTLR3_LIST of the strings that identify
     *  the rules in the parser that got you to this point. Can be overridden by installing your
     *	own function set.
     *
     * \todo Document how to override invocation stack functions.
     */
    pANTLR3_STACK	(*getRuleInvocationStack)	(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);
    pANTLR3_STACK	(*getRuleInvocationStackNamed)  (struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
								pANTLR3_UINT8	    name);

    /** Pointer to a function that converts an ANLR3_LIST of tokens to an ANTLR3_LIST of
     *  string token names. As this is mostly used in string template processing it may not be useful
     *  in the C runtime.
     */
    pANTLR3_HASH_TABLE	(*toStrings)			(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
								pANTLR3_HASH_TABLE);

    /** Pointer to a function to return whether the rule has parsed input starting at the supplied 
     *  start index before. If the rule has not parsed input starting from the supplied start index,
     *  then it will return ANTLR3_MEMO_RULE_UNKNOWN. If it has parsed from the suppled start point
     *  then it will return the point where it last stopped parsing after that start point.
     */
    ANTLR3_MARKER	(*getRuleMemoization)		(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
								ANTLR3_INTKEY	ruleIndex,
								ANTLR3_MARKER	ruleParseStart);

    /** Pointer to function that determines whether the rule has parsed input at the current index
     *  in the input stream
     */
    ANTLR3_BOOLEAN	(*alreadyParsedRule)		(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
								ANTLR3_MARKER	ruleIndex);

    /** Pointer to function that records whether the rule has parsed the input at a 
     *  current position successfully or not.
     */
    void		(*memoize)			(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
								ANTLR3_MARKER	ruleIndex,
								ANTLR3_MARKER	ruleParseStart);

	/// Pointer to a function that returns the current input symbol.
    /// The is placed into any label for the associated token ref; e.g., x=ID.  Token
	/// and tree parsers need to return different objects. Rather than test
	/// for input stream type or change the IntStream interface, I use
	/// a simple method to ask the recognizer to tell me what the current
	/// input symbol is.
	///
	/// This is ignored for lexers and the lexer implementation of this
	/// function should return NULL.
	///
	void *		(*getCurrentInputSymbol)	(	struct ANTLR3_BASE_RECOGNIZER_struct * recognizer, 
												pANTLR3_INT_STREAM istream);

	/// Conjure up a missing token during error recovery.
	///
	/// The recognizer attempts to recover from single missing
	/// symbols. But, actions might refer to that missing symbol.
	/// For example, x=ID {f($x);}. The action clearly assumes
	/// that there has been an identifier matched previously and that
	/// $x points at that token. If that token is missing, but
	/// the next token in the stream is what we want we assume that
	/// this token is missing and we keep going. Because we
	/// have to return some token to replace the missing token,
	/// we have to conjure one up. This method gives the user control
	/// over the tokens returned for missing tokens. Mostly,
	/// you will want to create something special for identifier
	/// tokens. For literals such as '{' and ',', the default
	/// action in the parser or tree parser works. It simply creates
	/// a CommonToken of the appropriate type. The text will be the token.
	/// If you change what tokens must be created by the lexer,
	/// override this method to create the appropriate tokens.
	///
	void *		(*getMissingSymbol)			(	struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,
												pANTLR3_INT_STREAM		istream,
												pANTLR3_EXCEPTION		e,
												ANTLR3_UINT32			expectedTokenType,
												pANTLR3_BITSET_LIST		follow);

    /** Pointer to a function that returns whether the supplied grammar function
     *  will parse the current input stream or not. This is the way that syntactic
     *  predicates are evaluated. Unlike java, C is perfectly happy to invoke code
     *  via a pointer to a function (hence that's what all the ANTLR3 C interfaces 
     *  do.
     */
    ANTLR3_BOOLEAN	(*synpred)			(	struct ANTLR3_BASE_RECOGNIZER_struct * recognizer,  void * ctx,
											void (*predicate)(void * ctx));

    /** Pointer to a function that can construct a generic exception structure
     * with such information as the input stream can provide.
     */
    void		    (*exConstruct)		(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Reset the recognizer
     */
    void		    (*reset)			(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

    /** Pointer to a function that knows how to free the resources of a base recognizer.
     */
    void			(*free)				(struct ANTLR3_BASE_RECOGNIZER_struct * recognizer);

}
    ANTLR3_BASE_RECOGNIZER;

#ifdef __cplusplus
}
#endif

#include    <antlr3lexer.h>
#include    <antlr3parser.h>
#include    <antlr3treeparser.h>

/// @}
///

#endif	    /* _ANTLR3_BASERECOGNIZER_H	*/

