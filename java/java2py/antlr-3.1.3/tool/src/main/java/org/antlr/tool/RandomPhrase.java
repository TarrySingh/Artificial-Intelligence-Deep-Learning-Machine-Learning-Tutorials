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

import org.antlr.analysis.*;
import org.antlr.misc.Utils;
import org.antlr.misc.IntervalSet;
import org.antlr.Tool;

import java.util.*;
import java.io.FileReader;
import java.io.BufferedReader;

/** Generate a random phrase given a grammar.
 *  Usage:
 *     java org.antlr.tool.RandomPhrase grammarFile.g startRule [seed]
 *
 *  For example:
 *     java org.antlr.tool.RandomPhrase simple.g program 342
 *
 *  The seed acts like a unique identifier so you can get the same random
 *  phrase back during unit testing, for example.
 *
 *  If you do not specify a seed then the current time in milliseconds is used
 *  guaranteeing that you'll never see that seed again.
 *
 *  NOTE: this does not work well for large grammars...it tends to recurse
 *  too much and build really long strings.  I need throttle control; later.
 */
public class RandomPhrase {
	public static final boolean debug = false;

	protected static Random random;

	/** an experimental method to generate random phrases for a given
	 *  grammar given a start rule.  Return a list of token types.
	 */
	protected static void randomPhrase(Grammar g, List<Integer> tokenTypes, String startRule) {
		NFAState state = g.getRuleStartState(startRule);
		NFAState stopState = g.getRuleStopState(startRule);

		Stack ruleInvocationStack = new Stack();
		while ( true ) {
			if ( state==stopState && ruleInvocationStack.size()==0 ) {
				break;
			}
			if ( debug ) System.out.println("state "+state);
			if ( state.getNumberOfTransitions()==0 ) {
				if ( debug ) System.out.println("dangling state: "+state);
				return;
			}
			// end of rule node
			if ( state.isAcceptState() ) {
				NFAState invokingState = (NFAState)ruleInvocationStack.pop();
				if ( debug ) System.out.println("pop invoking state "+invokingState);
				//System.out.println("leave "+state.enclosingRule.name);
				RuleClosureTransition invokingTransition =
					(RuleClosureTransition)invokingState.transition[0];
				// move to node after state that invoked this rule
				state = invokingTransition.followState;
				continue;
			}
			if ( state.getNumberOfTransitions()==1 ) {
				// no branching, just take this path
				Transition t0 = state.transition[0];
				if ( t0 instanceof RuleClosureTransition ) {
					ruleInvocationStack.push(state);
					if ( debug ) System.out.println("push state "+state);
					//System.out.println("call "+((RuleClosureTransition)t0).rule.name);
					//System.out.println("stack depth="+ruleInvocationStack.size());
				}
				else if ( t0.label.isSet() || t0.label.isAtom() ) {
					tokenTypes.add( getTokenType(t0.label) );
				}
				state = (NFAState)t0.target;
				continue;
			}

			int decisionNumber = state.getDecisionNumber();
			if ( decisionNumber==0 ) {
				System.out.println("weird: no decision number but a choice node");
				continue;
			}
			// decision point, pick ith alternative randomly
			int n = g.getNumberOfAltsForDecisionNFA(state);
			int randomAlt = random.nextInt(n) + 1;
			if ( debug ) System.out.println("randomAlt="+randomAlt);
			NFAState altStartState =
				g.getNFAStateForAltOfDecision(state, randomAlt);
			Transition t = altStartState.transition[0];
			state = (NFAState)t.target;
		}
	}

	protected static Integer getTokenType(Label label) {
		if ( label.isSet() ) {
			// pick random element of set
			IntervalSet typeSet = (IntervalSet)label.getSet();
			int randomIndex = random.nextInt(typeSet.size());
			return typeSet.get(randomIndex);
		}
		else {
			return Utils.integer(label.getAtom());
		}
		//System.out.println(t0.label.toString(g));
	}

	/** Used to generate random strings */
	public static void main(String[] args) {
		if ( args.length < 2 ) {
			System.err.println("usage: java org.antlr.tool.RandomPhrase grammarfile startrule");
			return;
		}
		String grammarFileName = args[0];
		String startRule = args[1];
		long seed = System.currentTimeMillis(); // use random seed unless spec.
		if ( args.length==3 ) {
			String seedStr = args[2];
			seed = Long.parseLong(seedStr);
		}
		try {
			random = new Random(seed);

			CompositeGrammar composite = new CompositeGrammar();
			Grammar parser = new Grammar(new Tool(), grammarFileName, composite);
			composite.setDelegationRoot(parser);

			FileReader fr = new FileReader(grammarFileName);
			BufferedReader br = new BufferedReader(fr);
			parser.parseAndBuildAST(br);
			br.close();

			parser.composite.assignTokenTypes();
			parser.composite.defineGrammarSymbols();
			parser.composite.createNFAs();

			List leftRecursiveRules = parser.checkAllRulesForLeftRecursion();
			if ( leftRecursiveRules.size()>0 ) {
				return;
			}

			if ( parser.getRule(startRule)==null ) {
				System.out.println("undefined start rule "+startRule);
				return;
			}

			String lexerGrammarText = parser.getLexerGrammar();
			Grammar lexer = new Grammar();
			lexer.importTokenVocabulary(parser);
			lexer.fileName = grammarFileName;
			if ( lexerGrammarText!=null ) {
				lexer.setGrammarContent(lexerGrammarText);
			}
			else {
				System.err.println("no lexer grammar found in "+grammarFileName);
			}
			lexer.buildNFA();
			leftRecursiveRules = lexer.checkAllRulesForLeftRecursion();
			if ( leftRecursiveRules.size()>0 ) {
				return;
			}
			//System.out.println("lexer:\n"+lexer);

			List<Integer> tokenTypes = new ArrayList<Integer>(100);
			randomPhrase(parser, tokenTypes, startRule);
			System.out.println("token types="+tokenTypes);
			for (int i = 0; i < tokenTypes.size(); i++) {
				Integer ttypeI = (Integer) tokenTypes.get(i);
				int ttype = ttypeI.intValue();
				String ttypeDisplayName = parser.getTokenDisplayName(ttype);
				if ( Character.isUpperCase(ttypeDisplayName.charAt(0)) ) {
					List<Integer> charsInToken = new ArrayList<Integer>(10);
					randomPhrase(lexer, charsInToken, ttypeDisplayName);
					System.out.print(" ");
					for (int j = 0; j < charsInToken.size(); j++) {
						java.lang.Integer cI = (java.lang.Integer) charsInToken.get(j);
						System.out.print((char)cI.intValue());
					}
				}
				else { // it's a literal
					String literal =
						ttypeDisplayName.substring(1,ttypeDisplayName.length()-1);
					System.out.print(" "+literal);
				}
			}
			System.out.println();
		}
		catch (Error er) {
			System.err.println("Error walking "+grammarFileName+" rule "+startRule+" seed "+seed);
			er.printStackTrace(System.err);
		}
		catch (Exception e) {
			System.err.println("Exception walking "+grammarFileName+" rule "+startRule+" seed "+seed);
			e.printStackTrace(System.err);
		}
	}
}
