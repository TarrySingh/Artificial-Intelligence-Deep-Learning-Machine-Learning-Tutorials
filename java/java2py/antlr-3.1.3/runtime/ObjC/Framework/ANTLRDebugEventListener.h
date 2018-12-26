// [The "BSD licence"]
// Copyright (c) 2006-2007 Kay Roepke
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

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLRToken.h>
#import <ANTLR/ANTLRRecognitionException.h>

@protocol ANTLRDebugEventListener 

#define ANTLRDebugProtocolVersion 1

/** The parser has just entered a rule.  No decision has been made about
*  which alt is predicted.  This is fired AFTER init actions have been
*  executed.  Attributes are defined and available etc...
*/
- (void) enterRule:(NSString *)ruleName;

/** Because rules can have lots of alternatives, it is very useful to
*  know which alt you are entering.  This is 1..n for n alts.
*/
- (void) enterAlt:(int)alt;

/** This is the last thing executed before leaving a rule.  It is
*  executed even if an exception is thrown.  This is triggered after
*  error reporting and recovery have occurred (unless the exception is
											   *  not caught in this rule).  This implies an "exitAlt" event.
*/
- (void) exitRule:(NSString *)ruleName;

/** Track entry into any (...) subrule other EBNF construct */
- (void) enterSubRule:(int)decisionNumber;

- (void) exitSubRule:(int)decisionNumber;

/** Every decision, fixed k or arbitrary, has an enter/exit event
*  so that a GUI can easily track what LT/consume events are
*  associated with prediction.  You will see a single enter/exit
*  subrule but multiple enter/exit decision events, one for each
*  loop iteration.
*/
- (void) enterDecision:(int)decisionNumber;

- (void) exitDecision:(int)decisionNumber;

/** An input token was consumed; matched by any kind of element.
*  Trigger after the token was matched by things like match(), matchAny().
*/
- (void) consumeToken:(id<ANTLRToken>)t;

/** An off-channel input token was consumed.
*  Trigger after the token was matched by things like match(), matchAny().
*  (unless of course the hidden token is first stuff in the input stream).
*/
- (void) consumeHiddenToken:(id<ANTLRToken>)t;

/** Somebody (anybody) looked ahead.  Note that this actually gets
*  triggered by both LA and LT calls.  The debugger will want to know
*  which Token object was examined.  Like consumeToken, this indicates
*  what token was seen at that depth.  A remote debugger cannot look
*  ahead into a file it doesn't have so LT events must pass the token
*  even if the info is redundant.
*/
- (void) LT:(int)i foundToken:(id<ANTLRToken>)t;

/** The parser is going to look arbitrarily ahead; mark this location,
*  the token stream's marker is sent in case you need it.
*/
- (void) mark:(int)marker;

/** After an arbitrairly long lookahead as with a cyclic DFA (or with
*  any backtrack), this informs the debugger that stream should be
*  rewound to the position associated with marker.
*/
- (void) rewind:(int)marker;

/** Rewind to the input position of the last marker.
*  Used currently only after a cyclic DFA and just
*  before starting a sem/syn predicate to get the
*  input position back to the start of the decision.
*  Do not "pop" the marker off the state.  mark(i)
*  and rewind(i) should balance still.
*/
- (void) rewind;

- (void) beginBacktrack:(int)level;

- (void) endBacktrack:(int)level wasSuccessful:(BOOL)successful;

/** To watch a parser move through the grammar, the parser needs to
*  inform the debugger what line/charPos it is passing in the grammar.
*  For now, this does not know how to switch from one grammar to the
*  other and back for island grammars etc...
*
*  This should also allow breakpoints because the debugger can stop
*  the parser whenever it hits this line/pos.
*/
- (void) locationLine:(int)line column:(int)pos;

/** A recognition exception occurred such as NoViableAltException.  I made
*  this a generic event so that I can alter the exception hierachy later
*  without having to alter all the debug objects.
*
*  Upon error, the stack of enter rule/subrule must be properly unwound.
*  If no viable alt occurs it is within an enter/exit decision, which
*  also must be rewound.  Even the rewind for each mark must be unwount.
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
- (void) recognitionException:(ANTLRRecognitionException *)e;

/** Indicates the recognizer is about to consume tokens to resynchronize
*  the parser.  Any consume events from here until the recovered event
*  are not part of the parse--they are dead tokens.
*/
- (void) beginResync;

/** Indicates that the recognizer has finished consuming tokens in order
*  to resychronize.  There may be multiple beginResync/endResync pairs
*  before the recognizer comes out of errorRecovery mode (in which
*  multiple errors are suppressed).  This will be useful
*  in a gui where you want to probably grey out tokens that are consumed
*  but not matched to anything in grammar.  Anything between
*  a beginResync/endResync pair was tossed out by the parser.
*/
- (void) endResync;

/** A semantic predicate was evaluate with this result and action text */
- (void) semanticPredicate:(NSString *)predicate matched:(BOOL)result;

/** Announce that parsing has begun.  Not technically useful except for
*  sending events over a socket.  A GUI for example will launch a thread
*  to connect and communicate with a remote parser.  The thread will want
*  to notify the GUI when a connection is made.  ANTLR parsers
*  trigger this upon entry to the first rule (the ruleLevel is used to
*  figure this out).
*/
- (void) commence;

/** Parsing is over; successfully or not.  Mostly useful for telling
*  remote debugging listeners that it's time to quit.  When the rule
*  invocation level goes to zero at the end of a rule, we are done
*  parsing.
*/
- (void) terminate;


// T r e e  P a r s i n g

/** Input for a tree parser is an AST, but we know nothing for sure
*  about a node except its type and text (obtained from the adaptor).
*  This is the analog of the consumeToken method.  Again, the ID is
*  the hashCode usually of the node so it only works if hashCode is
*  not implemented.  If the type is UP or DOWN, then
*  the ID is not really meaningful as it's fixed--there is
*  just one UP node and one DOWN navigation node.
*/
- (void) consumeNode:(int)nodeHash ofType:(int)type text:(NSString *)text;

/** The tree parser lookedahead.  If the type is UP or DOWN,
*  then the ID is not really meaningful as it's fixed--there is
*  just one UP node and one DOWN navigation node.
*/
- (void) LT:(int)i foundNode:(unsigned)nodeHash ofType:(int)type text:(NSString *)text;


// A S T  E v e n t s

/** A nil was created (even nil nodes have a unique ID...
*  they are not "null" per se).  As of 4/28/2006, this
*  seems to be uniquely triggered when starting a new subtree
*  such as when entering a subrule in automatic mode and when
*  building a tree in rewrite mode.
*/
- (void) createNilNode:(unsigned)hash;

/** Announce a new node built from text */
- (void) createNode:(unsigned)hash text:(NSString *)text type:(int)type;

/** Announce a new node built from an existing token */
- (void) createNode:(unsigned)hash fromTokenAtIndex:(int)tokenIndex;

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
*/
- (void) makeNode:(unsigned)newRootHash parentOf:(unsigned)oldRootHash;

/** Make childID a child of rootID.
*  @see org.antlr.runtime.tree.TreeAdaptor.addChild()
*/
- (void) addChild:(unsigned)childHash toTree:(unsigned)treeHash;

/** Set the token start/stop token index for a subtree root or node */
- (void) setTokenBoundariesForTree:(unsigned)nodeHash start:(unsigned int)tokenStartIndex stop:(unsigned int)tokenStopIndex;

- (void) waitForDebuggerConnection;

@end
