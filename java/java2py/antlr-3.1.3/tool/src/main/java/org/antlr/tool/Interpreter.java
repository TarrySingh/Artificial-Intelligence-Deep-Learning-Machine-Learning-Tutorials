/*
 [The "BSD licence"]
 Copyright (c) 2005-2008 Terence Parr
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
package org.antlr.tool;

import org.antlr.analysis.DFA;
import org.antlr.analysis.*;
import org.antlr.runtime.*;
import org.antlr.runtime.debug.DebugEventListener;
import org.antlr.runtime.debug.BlankDebugEventListener;
import org.antlr.runtime.tree.ParseTree;
import org.antlr.runtime.debug.ParseTreeBuilder;
import org.antlr.misc.IntervalSet;

import java.util.List;
import java.util.Stack;

/** The recognition interpreter/engine for grammars.  Separated
 *  out of Grammar as it's related, but technically not a Grammar function.
 *  You create an interpreter for a grammar and an input stream.  This object
 *  can act as a TokenSource so that you can hook up two grammars (via
 *  a CommonTokenStream) to lex/parse.  Being a token source only makes sense
 *  for a lexer grammar of course.
 */
public class Interpreter implements TokenSource {
	protected Grammar grammar;
	protected IntStream input;

	/** A lexer listener that just creates token objects as they
	 *  are matched.  scan() use this listener to get a single object.
	 *  To get a stream of tokens, you must call scan() multiple times,
	 *  recording the token object result after each call.
	 */
	class LexerActionGetTokenType extends BlankDebugEventListener {
		public CommonToken token;
		Grammar g;
		public LexerActionGetTokenType(Grammar g) {
			this.g = g;
		}

		public void exitRule(String grammarFileName, String ruleName) {
			if ( !ruleName.equals(Grammar.ARTIFICIAL_TOKENS_RULENAME) ){
				int type = g.getTokenType(ruleName);
				int channel = Token.DEFAULT_CHANNEL;
				token = new CommonToken((CharStream)input,type,channel,0,0);
			}
		}
	}

	public Interpreter(Grammar grammar, IntStream input) {
		this.grammar = grammar;
		this.input = input;
	}

	public Token nextToken() {
		if ( grammar.type!=Grammar.LEXER ) {
			return null;
		}
		if ( input.LA(1)==CharStream.EOF ) {
			return Token.EOF_TOKEN;
		}
		int start = input.index();
		int charPos = ((CharStream)input).getCharPositionInLine();
		CommonToken token = null;
		loop:
		while (input.LA(1)!=CharStream.EOF) {
			try {
				token = scan(Grammar.ARTIFICIAL_TOKENS_RULENAME, null);
				break;
			}
			catch (RecognitionException re) {
				// report a problem and try for another
				reportScanError(re);
				continue loop;
			}
		}
		// the scan can only set type
		// we must set the line, and other junk here to make it a complete token
		int stop = input.index()-1;
		if ( token==null ) {
			return Token.EOF_TOKEN;
		}
		token.setLine(((CharStream)input).getLine());
		token.setStartIndex(start);
		token.setStopIndex(stop);
		token.setCharPositionInLine(charPos);
		return token;
	}

	/** For a given input char stream, try to match against the NFA
	 *  starting at startRule.  This is a deterministic parse even though
	 *  it is using an NFA because it uses DFAs at each decision point to
	 *  predict which alternative will succeed.  This is exactly what the
	 *  generated parser will do.
	 *
	 *  This only does lexer grammars.
	 *
	 *  Return the token type associated with the final rule end state.
	 */
	public void scan(String startRule,
					 DebugEventListener actions,
					 List visitedStates)
		throws RecognitionException
	{
		if ( grammar.type!=Grammar.LEXER ) {
			return;
		}
		CharStream in = (CharStream)this.input;
		//System.out.println("scan("+startRule+",'"+in.substring(in.index(),in.size()-1)+"')");
		// Build NFAs/DFAs from the grammar AST if NFAs haven't been built yet
		if ( grammar.getRuleStartState(startRule)==null ) {
			grammar.buildNFA();
		}

		if ( !grammar.allDecisionDFAHaveBeenCreated() ) {
			// Create the DFA predictors for each decision
			grammar.createLookaheadDFAs();
		}

		// do the parse
		Stack ruleInvocationStack = new Stack();
		NFAState start = grammar.getRuleStartState(startRule);
		NFAState stop = grammar.getRuleStopState(startRule);
		parseEngine(startRule, start, stop, in, ruleInvocationStack,
					actions, visitedStates);
	}

	public CommonToken scan(String startRule)
		throws RecognitionException
	{
		return scan(startRule, null);
	}

	public CommonToken scan(String startRule,
							List visitedStates)
		throws RecognitionException
	{
		LexerActionGetTokenType actions = new LexerActionGetTokenType(grammar);
		scan(startRule, actions, visitedStates);
		return actions.token;
	}

	public void parse(String startRule,
					  DebugEventListener actions,
					  List visitedStates)
		throws RecognitionException
	{
		//System.out.println("parse("+startRule+")");
		// Build NFAs/DFAs from the grammar AST if NFAs haven't been built yet
		if ( grammar.getRuleStartState(startRule)==null ) {
			grammar.buildNFA();
		}
		if ( !grammar.allDecisionDFAHaveBeenCreated() ) {
			// Create the DFA predictors for each decision
			grammar.createLookaheadDFAs();
		}
		// do the parse
		Stack ruleInvocationStack = new Stack();
		NFAState start = grammar.getRuleStartState(startRule);
		NFAState stop = grammar.getRuleStopState(startRule);
		parseEngine(startRule, start, stop, input, ruleInvocationStack,
					actions, visitedStates);
	}

	public ParseTree parse(String startRule)
		throws RecognitionException
	{
		return parse(startRule, null);
	}

	public ParseTree parse(String startRule, List visitedStates)
		throws RecognitionException
	{
		ParseTreeBuilder actions = new ParseTreeBuilder(grammar.name);
		try {
			parse(startRule, actions, visitedStates);
		}
		catch (RecognitionException re) {
			// Errors are tracked via the ANTLRDebugInterface
			// Exceptions are used just to blast out of the parse engine
			// The error will be in the parse tree.
		}
		return actions.getTree();
	}

	/** Fill a list of all NFA states visited during the parse */
	protected void parseEngine(String startRule,
							   NFAState start,
							   NFAState stop,
							   IntStream input,
							   Stack ruleInvocationStack,
							   DebugEventListener actions,
							   List visitedStates)
		throws RecognitionException
	{
		NFAState s = start;
		if ( actions!=null ) {
			actions.enterRule(s.nfa.grammar.getFileName(), start.enclosingRule.name);
		}
		int t = input.LA(1);
		while ( s!=stop ) {
			if ( visitedStates!=null ) {
				visitedStates.add(s);
			}
			/*
			System.out.println("parse state "+s.stateNumber+" input="+
				s.nfa.grammar.getTokenDisplayName(t));
				*/
			// CASE 1: decision state
			if ( s.getDecisionNumber()>0 && s.nfa.grammar.getNumberOfAltsForDecisionNFA(s)>1 ) {
				// decision point, must predict and jump to alt
				DFA dfa = s.nfa.grammar.getLookaheadDFA(s.getDecisionNumber());
				/*
				if ( s.nfa.grammar.type!=Grammar.LEXER ) {
					System.out.println("decision: "+
								   dfa.getNFADecisionStartState().getDescription()+
								   " input="+s.nfa.grammar.getTokenDisplayName(t));
				}
				*/
				int m = input.mark();
				int predictedAlt = predict(dfa);
				if ( predictedAlt == NFA.INVALID_ALT_NUMBER ) {
					String description = dfa.getNFADecisionStartState().getDescription();
					NoViableAltException nvae =
						new NoViableAltException(description,
													  dfa.getDecisionNumber(),
													  s.stateNumber,
													  input);
					if ( actions!=null ) {
						actions.recognitionException(nvae);
					}
					input.consume(); // recover
					throw nvae;
				}
				input.rewind(m);
				int parseAlt =
					s.translateDisplayAltToWalkAlt(predictedAlt);
				/*
				if ( s.nfa.grammar.type!=Grammar.LEXER ) {
					System.out.println("predicted alt "+predictedAlt+", parseAlt "+
									   parseAlt);
				}
				*/
				NFAState alt;
				if ( parseAlt > s.nfa.grammar.getNumberOfAltsForDecisionNFA(s) ) {
					// implied branch of loop etc...
					alt = s.nfa.grammar.nfa.getState( s.endOfBlockStateNumber );
				}
				else {
					alt = s.nfa.grammar.getNFAStateForAltOfDecision(s, parseAlt);
				}
				s = (NFAState)alt.transition[0].target;
				continue;
			}

			// CASE 2: finished matching a rule
			if ( s.isAcceptState() ) { // end of rule node
				if ( actions!=null ) {
					actions.exitRule(s.nfa.grammar.getFileName(), s.enclosingRule.name);
				}
				if ( ruleInvocationStack.empty() ) {
					// done parsing.  Hit the start state.
					//System.out.println("stack empty in stop state for "+s.getEnclosingRule());
					break;
				}
				// pop invoking state off the stack to know where to return to
				NFAState invokingState = (NFAState)ruleInvocationStack.pop();
				RuleClosureTransition invokingTransition =
						(RuleClosureTransition)invokingState.transition[0];
				// move to node after state that invoked this rule
				s = invokingTransition.followState;
				continue;
			}

			Transition trans = s.transition[0];
			Label label = trans.label;
			if ( label.isSemanticPredicate() ) {
				FailedPredicateException fpe =
					new FailedPredicateException(input,
												 s.enclosingRule.name,
												 "can't deal with predicates yet");
				if ( actions!=null ) {
					actions.recognitionException(fpe);
				}
			}

			// CASE 3: epsilon transition
			if ( label.isEpsilon() ) {
				// CASE 3a: rule invocation state
				if ( trans instanceof RuleClosureTransition ) {
					ruleInvocationStack.push(s);
					s = (NFAState)trans.target;
					//System.out.println("call "+s.enclosingRule.name+" from "+s.nfa.grammar.getFileName());
					if ( actions!=null ) {
						actions.enterRule(s.nfa.grammar.getFileName(), s.enclosingRule.name);
					}
					// could be jumping to new grammar, make sure DFA created
					if ( !s.nfa.grammar.allDecisionDFAHaveBeenCreated() ) {
						s.nfa.grammar.createLookaheadDFAs();
					}
				}
				// CASE 3b: plain old epsilon transition, just move
				else {
					s = (NFAState)trans.target;
				}
			}

			// CASE 4: match label on transition
			else if ( label.matches(t) ) {
				if ( actions!=null ) {
					if ( s.nfa.grammar.type == Grammar.PARSER ||
						 s.nfa.grammar.type == Grammar.COMBINED )
					{
						actions.consumeToken(((TokenStream)input).LT(1));
					}
				}
				s = (NFAState)s.transition[0].target;
				input.consume();
				t = input.LA(1);
			}

			// CASE 5: error condition; label is inconsistent with input
			else {
				if ( label.isAtom() ) {
					MismatchedTokenException mte =
						new MismatchedTokenException(label.getAtom(), input);
					if ( actions!=null ) {
						actions.recognitionException(mte);
					}
					input.consume(); // recover
					throw mte;
				}
				else if ( label.isSet() ) {
					MismatchedSetException mse =
						new MismatchedSetException(((IntervalSet)label.getSet()).toRuntimeBitSet(),
												   input);
					if ( actions!=null ) {
						actions.recognitionException(mse);
					}
					input.consume(); // recover
					throw mse;
				}
				else if ( label.isSemanticPredicate() ) {
					FailedPredicateException fpe =
						new FailedPredicateException(input,
													 s.enclosingRule.name,
													 label.getSemanticContext().toString());
					if ( actions!=null ) {
						actions.recognitionException(fpe);
					}
					input.consume(); // recover
					throw fpe;
				}
				else {
					throw new RecognitionException(input); // unknown error
				}
			}
		}
		//System.out.println("hit stop state for "+stop.getEnclosingRule());
		if ( actions!=null ) {
			actions.exitRule(s.nfa.grammar.getFileName(), stop.enclosingRule.name);
		}
	}

	/** Given an input stream, return the unique alternative predicted by
	 *  matching the input.  Upon error, return NFA.INVALID_ALT_NUMBER
	 *  The first symbol of lookahead is presumed to be primed; that is,
	 *  input.lookahead(1) must point at the input symbol you want to start
	 *  predicting with.
	 */
	public int predict(DFA dfa) {
		DFAState s = dfa.startState;
		int c = input.LA(1);
		Transition eotTransition = null;
	dfaLoop:
		while ( !s.isAcceptState() ) {
			/*
			System.out.println("DFA.predict("+s.getStateNumber()+", "+
					dfa.getNFA().getGrammar().getTokenName(c)+")");
			*/
			// for each edge of s, look for intersection with current char
			for (int i=0; i<s.getNumberOfTransitions(); i++) {
				Transition t = s.transition(i);
				// special case: EOT matches any char
				if ( t.label.matches(c) ) {
					// take transition i
					s = (DFAState)t.target;
					input.consume();
					c = input.LA(1);
					continue dfaLoop;
				}
				if ( t.label.getAtom()==Label.EOT ) {
					eotTransition = t;
				}
			}
			if ( eotTransition!=null ) {
				s = (DFAState)eotTransition.target;
				continue dfaLoop;
			}
			/*
			ErrorManager.error(ErrorManager.MSG_NO_VIABLE_DFA_ALT,
							   s,
							   dfa.nfa.grammar.getTokenName(c));
			*/
			return NFA.INVALID_ALT_NUMBER;
		}
		// woohoo!  We know which alt to predict
		// nothing emanates from a stop state; must terminate anyway
		/*
		System.out.println("DFA stop state "+s.getStateNumber()+" predicts "+
				s.getUniquelyPredictedAlt());
		*/
		return s.getUniquelyPredictedAlt();
	}

	public void reportScanError(RecognitionException re) {
		CharStream cs = (CharStream)input;
		// print as good of a message as we can, given that we do not have
		// a Lexer object and, hence, cannot call the routine to get a
		// decent error message.
		System.err.println("problem matching token at "+
			cs.getLine()+":"+cs.getCharPositionInLine()+" "+re);
	}

	public String getSourceName() {
		return input.getSourceName();
	}

}
