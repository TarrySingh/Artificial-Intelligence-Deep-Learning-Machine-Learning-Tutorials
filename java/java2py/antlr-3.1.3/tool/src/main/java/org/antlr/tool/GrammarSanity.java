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

import org.antlr.analysis.NFAState;
import org.antlr.analysis.Transition;
import org.antlr.analysis.RuleClosureTransition;

import java.util.List;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Set;
import org.antlr.grammar.v2.ANTLRParser;

/** Factor out routines that check sanity of rules, alts, grammars, etc.. */
public class GrammarSanity {
	/** The checkForLeftRecursion method needs to track what rules it has
	 *  visited to track infinite recursion.
	 */
	protected Set<Rule> visitedDuringRecursionCheck = null;

	protected Grammar grammar;
	public GrammarSanity(Grammar grammar) {
		this.grammar = grammar;
	}

	/** Check all rules for infinite left recursion before analysis. Return list
	 *  of troublesome rule cycles.  This method has two side-effects: it notifies
	 *  the error manager that we have problems and it sets the list of
	 *  recursive rules that we should ignore during analysis.
	 */
	public List<Set<Rule>> checkAllRulesForLeftRecursion() {
		grammar.buildNFA(); // make sure we have NFAs
		grammar.leftRecursiveRules = new HashSet();
		List<Set<Rule>> listOfRecursiveCycles = new ArrayList();
		for (int i = 0; i < grammar.composite.ruleIndexToRuleList.size(); i++) {
			Rule r = grammar.composite.ruleIndexToRuleList.elementAt(i);
			if ( r!=null ) {
				visitedDuringRecursionCheck = new HashSet();
				visitedDuringRecursionCheck.add(r);
				Set visitedStates = new HashSet();
				traceStatesLookingForLeftRecursion(r.startState,
												   visitedStates,
												   listOfRecursiveCycles);
			}
		}
		if ( listOfRecursiveCycles.size()>0 ) {
			ErrorManager.leftRecursionCycles(listOfRecursiveCycles);
		}
		return listOfRecursiveCycles;
	}

	/** From state s, look for any transition to a rule that is currently
	 *  being traced.  When tracing r, visitedDuringRecursionCheck has r
	 *  initially.  If you reach an accept state, return but notify the
	 *  invoking rule that it is nullable, which implies that invoking
	 *  rule must look at follow transition for that invoking state.
	 *  The visitedStates tracks visited states within a single rule so
	 *  we can avoid epsilon-loop-induced infinite recursion here.  Keep
	 *  filling the cycles in listOfRecursiveCycles and also, as a
	 *  side-effect, set leftRecursiveRules.
	 */
	protected boolean traceStatesLookingForLeftRecursion(NFAState s,
														 Set visitedStates,
														 List<Set<Rule>> listOfRecursiveCycles)
	{
		if ( s.isAcceptState() ) {
			// this rule must be nullable!
			// At least one epsilon edge reached accept state
			return true;
		}
		if ( visitedStates.contains(s) ) {
			// within same rule, we've hit same state; quit looping
			return false;
		}
		visitedStates.add(s);
		boolean stateReachesAcceptState = false;
		Transition t0 = s.transition[0];
		if ( t0 instanceof RuleClosureTransition ) {
			RuleClosureTransition refTrans = (RuleClosureTransition)t0;
			Rule refRuleDef = refTrans.rule;
			//String targetRuleName = ((NFAState)t0.target).getEnclosingRule();
			if ( visitedDuringRecursionCheck.contains(refRuleDef) ) {
				// record left-recursive rule, but don't go back in
				grammar.leftRecursiveRules.add(refRuleDef);
				/*
				System.out.println("already visited "+refRuleDef+", calling from "+
								   s.enclosingRule);
								   */
				addRulesToCycle(refRuleDef,
								s.enclosingRule,
								listOfRecursiveCycles);
			}
			else {
				// must visit if not already visited; send new visitedStates set
				visitedDuringRecursionCheck.add(refRuleDef);
				boolean callReachedAcceptState =
					traceStatesLookingForLeftRecursion((NFAState)t0.target,
													   new HashSet(),
													   listOfRecursiveCycles);
				// we're back from visiting that rule
				visitedDuringRecursionCheck.remove(refRuleDef);
				// must keep going in this rule then
				if ( callReachedAcceptState ) {
					NFAState followingState =
						((RuleClosureTransition) t0).followState;
					stateReachesAcceptState |=
						traceStatesLookingForLeftRecursion(followingState,
														   visitedStates,
														   listOfRecursiveCycles);
				}
			}
		}
		else if ( t0.label.isEpsilon() || t0.label.isSemanticPredicate() ) {
			stateReachesAcceptState |=
				traceStatesLookingForLeftRecursion((NFAState)t0.target, visitedStates, listOfRecursiveCycles);
		}
		// else it has a labeled edge

		// now do the other transition if it exists
		Transition t1 = s.transition[1];
		if ( t1!=null ) {
			stateReachesAcceptState |=
				traceStatesLookingForLeftRecursion((NFAState)t1.target,
												   visitedStates,
												   listOfRecursiveCycles);
		}
		return stateReachesAcceptState;
	}

	/** enclosingRuleName calls targetRuleName, find the cycle containing
	 *  the target and add the caller.  Find the cycle containing the caller
	 *  and add the target.  If no cycles contain either, then create a new
	 *  cycle.  listOfRecursiveCycles is List<Set<String>> that holds a list
	 *  of cycles (sets of rule names).
	 */
	protected void addRulesToCycle(Rule targetRule,
								   Rule enclosingRule,
								   List<Set<Rule>> listOfRecursiveCycles)
	{
		boolean foundCycle = false;
		for (int i = 0; i < listOfRecursiveCycles.size(); i++) {
			Set<Rule> rulesInCycle = listOfRecursiveCycles.get(i);
			// ensure both rules are in same cycle
			if ( rulesInCycle.contains(targetRule) ) {
				rulesInCycle.add(enclosingRule);
				foundCycle = true;
			}
			if ( rulesInCycle.contains(enclosingRule) ) {
				rulesInCycle.add(targetRule);
				foundCycle = true;
			}
		}
		if ( !foundCycle ) {
			Set cycle = new HashSet();
			cycle.add(targetRule);
			cycle.add(enclosingRule);
			listOfRecursiveCycles.add(cycle);
		}
	}

	public void checkRuleReference(GrammarAST scopeAST,
								   GrammarAST refAST,
								   GrammarAST argsAST,
								   String currentRuleName)
	{
		Rule r = grammar.getRule(refAST.getText());
		if ( refAST.getType()==ANTLRParser.RULE_REF ) {
			if ( argsAST!=null ) {
				// rule[args]; ref has args
                if ( r!=null && r.argActionAST==null ) {
					// but rule def has no args
					ErrorManager.grammarError(
						ErrorManager.MSG_RULE_HAS_NO_ARGS,
						grammar,
						argsAST.getToken(),
						r.name);
				}
			}
			else {
				// rule ref has no args
				if ( r!=null && r.argActionAST!=null ) {
					// but rule def has args
					ErrorManager.grammarError(
						ErrorManager.MSG_MISSING_RULE_ARGS,
						grammar,
						refAST.getToken(),
						r.name);
				}
			}
		}
		else if ( refAST.getType()==ANTLRParser.TOKEN_REF ) {
			if ( grammar.type!=Grammar.LEXER ) {
				if ( argsAST!=null ) {
					// args on a token ref not in a lexer rule
					ErrorManager.grammarError(
						ErrorManager.MSG_ARGS_ON_TOKEN_REF,
						grammar,
						refAST.getToken(),
						refAST.getText());
				}
				return; // ignore token refs in nonlexers
			}
			if ( argsAST!=null ) {
				// tokenRef[args]; ref has args
				if ( r!=null && r.argActionAST==null ) {
					// but token rule def has no args
					ErrorManager.grammarError(
						ErrorManager.MSG_RULE_HAS_NO_ARGS,
						grammar,
						argsAST.getToken(),
						r.name);
				}
			}
			else {
				// token ref has no args
				if ( r!=null && r.argActionAST!=null ) {
					// but token rule def has args
					ErrorManager.grammarError(
						ErrorManager.MSG_MISSING_RULE_ARGS,
						grammar,
						refAST.getToken(),
						r.name);
				}
			}
		}
	}

	/** Rules in tree grammar that use -> rewrites and are spitting out
	 *  templates via output=template and then use rewrite=true must only
	 *  use -> on alts that are simple nodes or trees or single rule refs
	 *  that match either nodes or trees.  The altAST is the ALT node
	 *  for an ALT.  Verify that its first child is simple.  Must be either
	 *  ( ALT ^( A B ) <end-of-alt> ) or ( ALT A <end-of-alt> ) or
	 *  other element.
	 *
	 *  Ignore predicates in front and labels.
	 */
	public void ensureAltIsSimpleNodeOrTree(GrammarAST altAST,
											GrammarAST elementAST,
											int outerAltNum)
	{
		if ( isValidSimpleElementNode(elementAST) ) {
			GrammarAST next = (GrammarAST)elementAST.getNextSibling();
			if ( !isNextNonActionElementEOA(next)) {
				ErrorManager.grammarWarning(ErrorManager.MSG_REWRITE_FOR_MULTI_ELEMENT_ALT,
											grammar,
											next.token,
											new Integer(outerAltNum));
			}
			return;
		}
		switch ( elementAST.getType() ) {
			case ANTLRParser.ASSIGN :		// labels ok on non-rule refs
			case ANTLRParser.PLUS_ASSIGN :
				if ( isValidSimpleElementNode(elementAST.getChild(1)) ) {
					return;
				}
				break;
			case ANTLRParser.ACTION :		// skip past actions
			case ANTLRParser.SEMPRED :
			case ANTLRParser.SYN_SEMPRED :
			case ANTLRParser.BACKTRACK_SEMPRED :
			case ANTLRParser.GATED_SEMPRED :
				ensureAltIsSimpleNodeOrTree(altAST,
											(GrammarAST)elementAST.getNextSibling(),
											outerAltNum);
				return;
		}
		ErrorManager.grammarWarning(ErrorManager.MSG_REWRITE_FOR_MULTI_ELEMENT_ALT,
									grammar,
									elementAST.token,
									new Integer(outerAltNum));
	}

	protected boolean isValidSimpleElementNode(GrammarAST t) {
		switch ( t.getType() ) {
			case ANTLRParser.TREE_BEGIN :
			case ANTLRParser.TOKEN_REF :
			case ANTLRParser.CHAR_LITERAL :
			case ANTLRParser.STRING_LITERAL :
			case ANTLRParser.WILDCARD :
				return true;
			default :
				return false;
		}
	}

	protected boolean isNextNonActionElementEOA(GrammarAST t) {
		while ( t.getType()==ANTLRParser.ACTION ||
				t.getType()==ANTLRParser.SEMPRED )
		{
			t = (GrammarAST)t.getNextSibling();
		}
		if ( t.getType()==ANTLRParser.EOA ) {
			return true;
		}
		return false;
	}
}
