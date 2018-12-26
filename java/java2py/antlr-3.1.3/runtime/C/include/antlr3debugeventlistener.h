/**
 * \file
 * The definition of all debugging events that a recognizer can trigger.
 *
 * \remark
 *  From the java implementation by Terence Parr...
 *  I did not create a separate AST debugging interface as it would create
 *  lots of extra classes and DebugParser has a dbg var defined, which makes
 *  it hard to change to ASTDebugEventListener.  I looked hard at this issue
 *  and it is easier to understand as one monolithic event interface for all
 *  possible events.  Hopefully, adding ST debugging stuff won't be bad.  Leave
 *  for future. 4/26/2006.
 */

#ifndef	ANTLR3_DEBUG_EVENT_LISTENER_H
#define	ANTLR3_DEBUG_EVENT_LISTENER_H

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
#include    <antlr3basetree.h>
#include    <antlr3commontoken.h>


/// Default debugging port
///
#define DEFAULT_DEBUGGER_PORT		0xBFCC;

#ifdef __cplusplus
extern "C" {
#endif

/** The ANTLR3 debugging interface for communicating with ANLTR Works. Function comments
 *  mostly taken from the Java version.
 */
typedef struct ANTLR3_DEBUG_EVENT_LISTENER_struct
{
	/// The port number which the debug listener should listen on for a connection
	///
	ANTLR3_UINT32		port;

	/// The socket structure we receive after a successful accept on the serverSocket
	///
	SOCKET				socket;

	/** The version of the debugging protocol supported by the providing
	 *  instance of the debug event listener.
	 */
	int					PROTOCOL_VERSION;

	/// The name of the grammar file that we are debugging
	///
	pANTLR3_STRING		grammarFileName;

	/// Indicates whether we have already connected or not
	///
	ANTLR3_BOOLEAN		initialized;

	/// Used to serialize the values of any particular token we need to
	/// send back to the debugger.
	///
	pANTLR3_STRING		tokenString;


	/// Allows the debug event system to access the adapter in use
	/// by the recognizer, if this is a tree parser of some sort.
	///
	pANTLR3_BASE_TREE_ADAPTOR	adaptor;

	/// Wait for a connection from the debugger and initiate the
	/// debugging session.
	///
	ANTLR3_BOOLEAN	(*handshake)		(pANTLR3_DEBUG_EVENT_LISTENER delboy);

	/** The parser has just entered a rule.  No decision has been made about
	 *  which alt is predicted.  This is fired AFTER init actions have been
	 *  executed.  Attributes are defined and available etc...
	 */
	void			(*enterRule)		(pANTLR3_DEBUG_EVENT_LISTENER delboy, const char * grammarFileName, const char * ruleName);

	/** Because rules can have lots of alternatives, it is very useful to
	 *  know which alt you are entering.  This is 1..n for n alts.
	 */
	void			(*enterAlt)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int alt);

	/** This is the last thing executed before leaving a rule.  It is
	 *  executed even if an exception is thrown.  This is triggered after
	 *  error reporting and recovery have occurred (unless the exception is
	 *  not caught in this rule).  This implies an "exitAlt" event.
	 */
	void			(*exitRule)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, const char * grammarFileName, const char * ruleName);

	/** Track entry into any (...) subrule other EBNF construct 
	 */
	void			(*enterSubRule)		(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);
	
	void			(*exitSubRule)		(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);

	/** Every decision, fixed k or arbitrary, has an enter/exit event
	 *  so that a GUI can easily track what LT/consume events are
	 *  associated with prediction.  You will see a single enter/exit
	 *  subrule but multiple enter/exit decision events, one for each
	 *  loop iteration.
	 */
	void			(*enterDecision)	(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);

	void			(*exitDecision)		(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);

	/** An input token was consumed; matched by any kind of element.
	 *  Trigger after the token was matched by things like match(), matchAny().
	 */
	void			(*consumeToken)		(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_COMMON_TOKEN t);

	/** An off-channel input token was consumed.
	 *  Trigger after the token was matched by things like match(), matchAny().
	 *  (unless of course the hidden token is first stuff in the input stream).
	 */
	void			(*consumeHiddenToken)	(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_COMMON_TOKEN t);

	/** Somebody (anybody) looked ahead.  Note that this actually gets
	 *  triggered by both LA and LT calls.  The debugger will want to know
	 *  which Token object was examined.  Like consumeToken, this indicates
	 *  what token was seen at that depth.  A remote debugger cannot look
	 *  ahead into a file it doesn't have so LT events must pass the token
	 *  even if the info is redundant.
	 */
	void			(*LT)				(pANTLR3_DEBUG_EVENT_LISTENER delboy, int i, pANTLR3_COMMON_TOKEN t);

	/** The parser is going to look arbitrarily ahead; mark this location,
	 *  the token stream's marker is sent in case you need it.
	 */
	void			(*mark)				(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_MARKER marker);

	/** After an arbitrarily long lookahead as with a cyclic DFA (or with
	 *  any backtrack), this informs the debugger that stream should be
	 *  rewound to the position associated with marker.
	 */
	void			(*rewind)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_MARKER marker);

	/** Rewind to the input position of the last marker.
	 *  Used currently only after a cyclic DFA and just
	 *  before starting a sem/syn predicate to get the
	 *  input position back to the start of the decision.
	 *  Do not "pop" the marker off the state.  mark(i)
	 *  and rewind(i) should balance still.
	 */
	void			(*rewindLast)		(pANTLR3_DEBUG_EVENT_LISTENER delboy);

	void			(*beginBacktrack)	(pANTLR3_DEBUG_EVENT_LISTENER delboy, int level);

	void			(*endBacktrack)		(pANTLR3_DEBUG_EVENT_LISTENER delboy, int level, ANTLR3_BOOLEAN successful);

	/** To watch a parser move through the grammar, the parser needs to
	 *  inform the debugger what line/charPos it is passing in the grammar.
	 *  For now, this does not know how to switch from one grammar to the
	 *  other and back for island grammars etc...
	 *
	 *  This should also allow breakpoints because the debugger can stop
	 *  the parser whenever it hits this line/pos.
	 */
	void			(*location)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int line, int pos);

	/** A recognition exception occurred such as NoViableAltException.  I made
	 *  this a generic event so that I can alter the exception hierarchy later
	 *  without having to alter all the debug objects.
	 *
	 *  Upon error, the stack of enter rule/subrule must be properly unwound.
	 *  If no viable alt occurs it is within an enter/exit decision, which
	 *  also must be rewound.  Even the rewind for each mark must be unwound.
	 *  In the Java target this is pretty easy using try/finally, if a bit
	 *  ugly in the generated code.  The rewind is generated in DFA.predict()
	 *  actually so no code needs to be generated for that.  For languages
	 *  w/o this "finally" feature (C++?), the target implementor will have
	 *  to build an event stack or something.
	 *
	 *  Across a socket for remote debugging, only the RecognitionException
	 *  data fields are transmitted.  The token object or whatever that
	 *  caused the problem was the last object referenced by LT.  The
	 *  immediately preceding LT event should hold the unexpected Token or
	 *  char.
	 *
	 *  Here is a sample event trace for grammar:
	 *
	 *  b : C ({;}A|B) // {;} is there to prevent A|B becoming a set
     *    | D
     *    ;
     *
	 *  The sequence for this rule (with no viable alt in the subrule) for
	 *  input 'c c' (there are 3 tokens) is:
	 *
	 *		commence
	 *		LT(1)
	 *		enterRule b
	 *		location 7 1
	 *		enter decision 3
	 *		LT(1)
	 *		exit decision 3
	 *		enterAlt1
	 *		location 7 5
	 *		LT(1)
	 *		consumeToken [c/<4>,1:0]
	 *		location 7 7
	 *		enterSubRule 2
	 *		enter decision 2
	 *		LT(1)
	 *		LT(1)
	 *		recognitionException NoViableAltException 2 1 2
	 *		exit decision 2
	 *		exitSubRule 2
	 *		beginResync
	 *		LT(1)
	 *		consumeToken [c/<4>,1:1]
	 *		LT(1)
	 *		endResync
	 *		LT(-1)
	 *		exitRule b
	 *		terminate
	 */
	void			(*recognitionException)	(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_EXCEPTION e);

	/** Indicates the recognizer is about to consume tokens to resynchronize
	 *  the parser.  Any consume events from here until the recovered event
	 *  are not part of the parse--they are dead tokens.
	 */
	void			(*beginResync)			(pANTLR3_DEBUG_EVENT_LISTENER delboy);

	/** Indicates that the recognizer has finished consuming tokens in order
	 *  to resynchronize.  There may be multiple beginResync/endResync pairs
	 *  before the recognizer comes out of errorRecovery mode (in which
	 *  multiple errors are suppressed).  This will be useful
	 *  in a gui where you want to probably grey out tokens that are consumed
	 *  but not matched to anything in grammar.  Anything between
	 *  a beginResync/endResync pair was tossed out by the parser.
	 */
	void			(*endResync)			(pANTLR3_DEBUG_EVENT_LISTENER delboy);

	/** A semantic predicate was evaluate with this result and action text 
	*/
	void			(*semanticPredicate)	(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_BOOLEAN result, const char * predicate);

	/** Announce that parsing has begun.  Not technically useful except for
	 *  sending events over a socket.  A GUI for example will launch a thread
	 *  to connect and communicate with a remote parser.  The thread will want
	 *  to notify the GUI when a connection is made.  ANTLR parsers
	 *  trigger this upon entry to the first rule (the ruleLevel is used to
	 *  figure this out).
	 */
	void			(*commence)				(pANTLR3_DEBUG_EVENT_LISTENER delboy);

	/** Parsing is over; successfully or not.  Mostly useful for telling
	 *  remote debugging listeners that it's time to quit.  When the rule
	 *  invocation level goes to zero at the end of a rule, we are done
	 *  parsing.
	 */
	void			(*terminate)			(pANTLR3_DEBUG_EVENT_LISTENER delboy);

	/// Retrieve acknowledge response from the debugger. in fact this
	/// response is never used at the moment. So we just read whatever
	/// is in the socket buffer and throw it away.
	///
	void			(*ack)					(pANTLR3_DEBUG_EVENT_LISTENER delboy);

	// T r e e  P a r s i n g

	/** Input for a tree parser is an AST, but we know nothing for sure
	 *  about a node except its type and text (obtained from the adaptor).
	 *  This is the analog of the consumeToken method.  The ID is usually 
	 *  the memory address of the node.
	 *  If the type is UP or DOWN, then
	 *  the ID is not really meaningful as it's fixed--there is
	 *  just one UP node and one DOWN navigation node.
	 *
	 *  Note that unlike the Java version, the node type of the C parsers
	 *  is always fixed as pANTLR3_BASE_TREE because all such structures
	 *  contain a super pointer to their parent, which is generally COMMON_TREE and within
	 *  that there is a super pointer that can point to a user type that encapsulates it.
	 *  Almost akin to saying that it is an interface pointer except we don't need to
	 *  know what the interface is in full, just those bits that are the base.
	 * @param t
	 */
	void			(*consumeNode)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);

	/** The tree parser looked ahead.  If the type is UP or DOWN,
	 *  then the ID is not really meaningful as it's fixed--there is
	 *  just one UP node and one DOWN navigation node.
	 */
	void			(*LTT)					(pANTLR3_DEBUG_EVENT_LISTENER delboy, int i, pANTLR3_BASE_TREE t);


	// A S T  E v e n t s

	/** A nil was created (even nil nodes have a unique ID...
	 *  they are not "null" per se).  As of 4/28/2006, this
	 *  seems to be uniquely triggered when starting a new subtree
	 *  such as when entering a subrule in automatic mode and when
	 *  building a tree in rewrite mode.
     *
 	 *  If you are receiving this event over a socket via
	 *  RemoteDebugEventSocketListener then only t.ID is set.
	 */
	void			(*nilNode)				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);

	/** If a syntax error occurs, recognizers bracket the error
	 *  with an error node if they are building ASTs. This event
	 *  notifies the listener that this is the case
	 */
	void			(*errorNode)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);

	/** Announce a new node built from token elements such as type etc...
	 * 
	 *  If you are receiving this event over a socket via
	 *  RemoteDebugEventSocketListener then only t.ID, type, text are
	 *  set.
	 */
	void			(*createNode)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);

	/** Announce a new node built from an existing token.
	 *
	 *  If you are receiving this event over a socket via
	 *  RemoteDebugEventSocketListener then only node.ID and token.tokenIndex
	 *  are set.
	 */
	void			(*createNodeTok)		(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE node, pANTLR3_COMMON_TOKEN token);

	/** Make a node the new root of an existing root.  See
	 *
	 *  Note: the newRootID parameter is possibly different
	 *  than the TreeAdaptor.becomeRoot() newRoot parameter.
	 *  In our case, it will always be the result of calling
	 *  TreeAdaptor.becomeRoot() and not root_n or whatever.
	 *
	 *  The listener should assume that this event occurs
	 *  only when the current subrule (or rule) subtree is
	 *  being reset to newRootID.
	 * 
	 *  If you are receiving this event over a socket via
	 *  RemoteDebugEventSocketListener then only IDs are set.
	 *
	 *  @see org.antlr.runtime.tree.TreeAdaptor.becomeRoot()
	 */
	void			(*becomeRoot)			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE newRoot, pANTLR3_BASE_TREE oldRoot);

	/** Make childID a child of rootID.
	 *
	 *  If you are receiving this event over a socket via
	 *  RemoteDebugEventSocketListener then only IDs are set.
	 * 
	 *  @see org.antlr.runtime.tree.TreeAdaptor.addChild()
	 */
	void			(*addChild)				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE root, pANTLR3_BASE_TREE child);

	/** Set the token start/stop token index for a subtree root or node.
	 *
	 *  If you are receiving this event over a socket via
	 *  RemoteDebugEventSocketListener then only t.ID is set.
	 */
	void			(*setTokenBoundaries)	(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t, ANTLR3_MARKER tokenStartIndex, ANTLR3_MARKER tokenStopIndex);

	/// Free up the resources allocated to this structure
	///
	void			(*free)					(pANTLR3_DEBUG_EVENT_LISTENER delboy);

}
	ANTLR3_DEBUG_EVENT_LISTENER;

#ifdef __cplusplus
}
#endif

#endif

