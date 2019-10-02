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
package org.antlr.analysis;

import org.antlr.misc.IntervalSet;
import org.antlr.misc.MultiMap;
import org.antlr.grammar.v2.ANTLRParser;

import java.util.Iterator;
import java.util.List;
import java.util.Collections;

/** A special DFA that is exactly LL(1) or LL(1) with backtracking mode
 *  predicates to resolve edge set collisions.
 */
public class LL1DFA extends DFA {
	/** From list of lookahead sets (one per alt in decision), create
	 *  an LL(1) DFA.  One edge per set.
	 *
	 *  s0-{alt1}->:o=>1
	 *  | \
	 *  |  -{alt2}->:o=>2
	 *  |
	 *  ...
	 */
	public LL1DFA(int decisionNumber, NFAState decisionStartState, LookaheadSet[] altLook) {
		DFAState s0 = newState();
		startState = s0;
		nfa = decisionStartState.nfa;
		nAlts = nfa.grammar.getNumberOfAltsForDecisionNFA(decisionStartState);
		this.decisionNumber = decisionNumber;
		this.decisionNFAStartState = decisionStartState;
		initAltRelatedInfo();
		unreachableAlts = null;
		for (int alt=1; alt<altLook.length; alt++) {
			DFAState acceptAltState = newState();
			acceptAltState.acceptState = true;
			setAcceptState(alt, acceptAltState);
			acceptAltState.k = 1;
			acceptAltState.cachedUniquelyPredicatedAlt = alt;
			Label e = getLabelForSet(altLook[alt].tokenTypeSet);
			s0.addTransition(acceptAltState, e);
		}
	}

	/** From a set of edgeset->list-of-alts mappings, create a DFA
	 *  that uses syn preds for all |list-of-alts|>1.
	 */
	public LL1DFA(int decisionNumber,
				  NFAState decisionStartState,
				  MultiMap<IntervalSet, Integer> edgeMap)
	{
		DFAState s0 = newState();
		startState = s0;
		nfa = decisionStartState.nfa;
		nAlts = nfa.grammar.getNumberOfAltsForDecisionNFA(decisionStartState);
		this.decisionNumber = decisionNumber;
		this.decisionNFAStartState = decisionStartState;
		initAltRelatedInfo();
		unreachableAlts = null;
		for (Iterator it = edgeMap.keySet().iterator(); it.hasNext();) {
			IntervalSet edge = (IntervalSet)it.next();
			List<Integer> alts = edgeMap.get(edge);
			Collections.sort(alts); // make sure alts are attempted in order
			//System.out.println(edge+" -> "+alts);
			DFAState s = newState();
			s.k = 1;
			Label e = getLabelForSet(edge);
			s0.addTransition(s, e);
			if ( alts.size()==1 ) {
				s.acceptState = true;
				int alt = alts.get(0);
				setAcceptState(alt, s);
				s.cachedUniquelyPredicatedAlt = alt;
			}
			else {
				// resolve with syntactic predicates.  Add edges from
				// state s that test predicates.
				s.resolvedWithPredicates = true;
				for (int i = 0; i < alts.size(); i++) {
					int alt = (int)alts.get(i);
					s.cachedUniquelyPredicatedAlt =	NFA.INVALID_ALT_NUMBER;
					DFAState predDFATarget = getAcceptState(alt);
					if ( predDFATarget==null ) {
						predDFATarget = newState(); // create if not there.
						predDFATarget.acceptState = true;
						predDFATarget.cachedUniquelyPredicatedAlt =	alt;
						setAcceptState(alt, predDFATarget);
					}
					// add a transition to pred target from d
					/*
					int walkAlt =
						decisionStartState.translateDisplayAltToWalkAlt(alt);
					NFAState altLeftEdge = nfa.grammar.getNFAStateForAltOfDecision(decisionStartState, walkAlt);
					NFAState altStartState = (NFAState)altLeftEdge.transition[0].target;
					SemanticContext ctx = nfa.grammar.ll1Analyzer.getPredicates(altStartState);
					System.out.println("sem ctx = "+ctx);
					if ( ctx == null ) {
						ctx = new SemanticContext.TruePredicate();
					}
					s.addTransition(predDFATarget, new Label(ctx));
					*/
					SemanticContext.Predicate synpred =
						getSynPredForAlt(decisionStartState, alt);
					if ( synpred == null ) {
						synpred = new SemanticContext.TruePredicate();
					}
					s.addTransition(predDFATarget, new PredicateLabel(synpred));
				}
			}
		}
		//System.out.println("dfa for preds=\n"+this);
	}

	protected Label getLabelForSet(IntervalSet edgeSet) {
		Label e = null;
		int atom = edgeSet.getSingleElement();
		if ( atom != Label.INVALID ) {
			e = new Label(atom);
		}
		else {
			e = new Label(edgeSet);
		}
		return e;
	}

	protected SemanticContext.Predicate getSynPredForAlt(NFAState decisionStartState,
														 int alt)
	{
		int walkAlt =
			decisionStartState.translateDisplayAltToWalkAlt(alt);
		NFAState altLeftEdge =
			nfa.grammar.getNFAStateForAltOfDecision(decisionStartState, walkAlt);
		NFAState altStartState = (NFAState)altLeftEdge.transition[0].target;
		//System.out.println("alt "+alt+" start state = "+altStartState.stateNumber);
		if ( altStartState.transition[0].isSemanticPredicate() ) {
			SemanticContext ctx = altStartState.transition[0].label.getSemanticContext();
			if ( ctx.isSyntacticPredicate() ) {
				SemanticContext.Predicate p = (SemanticContext.Predicate)ctx;
				if ( p.predicateAST.getType() == ANTLRParser.BACKTRACK_SEMPRED ) {
					/*
					System.out.println("syn pred for alt "+walkAlt+" "+
									   ((SemanticContext.Predicate)altStartState.transition[0].label.getSemanticContext()).predicateAST);
					*/
					if ( ctx.isSyntacticPredicate() ) {
						nfa.grammar.synPredUsedInDFA(this, ctx);
					}
					return (SemanticContext.Predicate)altStartState.transition[0].label.getSemanticContext();
				}
			}
		}
		return null;
	}
}
