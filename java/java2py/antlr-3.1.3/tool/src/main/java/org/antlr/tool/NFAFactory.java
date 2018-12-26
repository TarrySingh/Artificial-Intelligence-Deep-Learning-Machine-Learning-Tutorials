/*
 [The "BSD licence"]
 Copyright (c) 2005-2006 Terence Parr
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
import org.antlr.misc.IntSet;
import org.antlr.misc.IntervalSet;

import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;

import antlr.Token;

/** Routines to construct StateClusters from EBNF grammar constructs.
 *  No optimization is done to remove unnecessary epsilon edges.
 *
 *  TODO: add an optimization that reduces number of states and transitions
 *  will help with speed of conversion and make it easier to view NFA.  For
 *  example, o-A->o-->o-B->o should be o-A->o-B->o
 */
public class NFAFactory {
	/** This factory is attached to a specifc NFA that it is building.
     *  The NFA will be filled up with states and transitions.
     */
	NFA nfa = null;

    public Rule getCurrentRule() {
        return currentRule;
    }

    public void setCurrentRule(Rule currentRule) {
        this.currentRule = currentRule;
    }

	Rule currentRule = null;

	public NFAFactory(NFA nfa) {
        nfa.setFactory(this);
		this.nfa = nfa;
	}

    public NFAState newState() {
        NFAState n = new NFAState(nfa);
        int state = nfa.getNewNFAStateNumber();
        n.stateNumber = state;
        nfa.addState(n);
		n.enclosingRule = currentRule;
		return n;
    }

	/** Optimize an alternative (list of grammar elements).
	 *
	 *  Walk the chain of elements (which can be complicated loop blocks...)
	 *  and throw away any epsilon transitions used to link up simple elements.
	 *
	 *  This only removes 195 states from the java.g's NFA, but every little
	 *  bit helps.  Perhaps I can improve in the future.
	 */
	public void optimizeAlternative(StateCluster alt) {
		NFAState s = alt.left;
		while ( s!=alt.right ) {
			// if it's a block element, jump over it and continue
			if ( s.endOfBlockStateNumber!=State.INVALID_STATE_NUMBER ) {
				s = nfa.getState(s.endOfBlockStateNumber);
				continue;
			}
			Transition t = s.transition[0];
			if ( t instanceof RuleClosureTransition ) {
				s = ((RuleClosureTransition) t).followState;
				continue;
			}
			if ( t.label.isEpsilon() && !t.label.isAction() && s.getNumberOfTransitions()==1 ) {
				// bypass epsilon transition and point to what the epsilon's
				// target points to unless that epsilon transition points to
				// a block or loop etc..  Also don't collapse epsilons that
				// point at the last node of the alt. Don't collapse action edges
				NFAState epsilonTarget = (NFAState)t.target;
				if ( epsilonTarget.endOfBlockStateNumber==State.INVALID_STATE_NUMBER &&
					 epsilonTarget.transition[0] !=null )
				{
					s.setTransition0(epsilonTarget.transition[0]);
					/*
					System.out.println("### opt "+s.stateNumber+"->"+
									   epsilonTarget.transition(0).target.stateNumber);
					*/
				}
			}
			s = (NFAState)t.target;
		}
	}

	/** From label A build Graph o-A->o */
	public StateCluster build_Atom(int label, GrammarAST associatedAST) {
		NFAState left = newState();
		NFAState right = newState();
		left.associatedASTNode = associatedAST;
		right.associatedASTNode = associatedAST;
		transitionBetweenStates(left, right, label);
		StateCluster g = new StateCluster(left, right);
		return g;
	}

	public StateCluster build_Atom(GrammarAST atomAST) {
		int tokenType = nfa.grammar.getTokenType(atomAST.getText());
		return build_Atom(tokenType, atomAST);
	}

	/** From set build single edge graph o->o-set->o.  To conform to
     *  what an alt block looks like, must have extra state on left.
     */
	public StateCluster build_Set(IntSet set, GrammarAST associatedAST) {
        NFAState left = newState();
        NFAState right = newState();
		left.associatedASTNode = associatedAST;
		right.associatedASTNode = associatedAST;
		Label label = new Label(set);
		Transition e = new Transition(label,right);
        left.addTransition(e);
		StateCluster g = new StateCluster(left, right);
        return g;
	}

    /** Can only complement block of simple alts; can complement build_Set()
     *  result, that is.  Get set and complement, replace old with complement.
    public StateCluster build_AlternativeBlockComplement(StateCluster blk) {
        State s0 = blk.left;
        IntSet set = getCollapsedBlockAsSet(s0);
        if ( set!=null ) {
            // if set is available, then structure known and blk is a set
            set = nfa.grammar.complement(set);
            Label label = s0.transition(0).target.transition(0).label;
            label.setSet(set);
        }
        return blk;
    }
	 */

    public StateCluster build_Range(int a, int b) {
        NFAState left = newState();
        NFAState right = newState();
		Label label = new Label(IntervalSet.of(a, b));
		Transition e = new Transition(label,right);
        left.addTransition(e);
        StateCluster g = new StateCluster(left, right);
        return g;
    }

	/** From char 'c' build StateCluster o-intValue(c)->o
	 */
	public StateCluster build_CharLiteralAtom(GrammarAST charLiteralAST) {
        int c = Grammar.getCharValueFromGrammarCharLiteral(charLiteralAST.getText());
		return build_Atom(c, charLiteralAST);
	}

	/** From char 'c' build StateCluster o-intValue(c)->o
	 *  can include unicode spec likes '\u0024' later.  Accepts
	 *  actual unicode 16-bit now, of course, by default.
     *  TODO not supplemental char clean!
	 */
	public StateCluster build_CharRange(String a, String b) {
		int from = Grammar.getCharValueFromGrammarCharLiteral(a);
		int to = Grammar.getCharValueFromGrammarCharLiteral(b);
		return build_Range(from, to);
	}

    /** For a non-lexer, just build a simple token reference atom.
     *  For a lexer, a string is a sequence of char to match.  That is,
     *  "fog" is treated as 'f' 'o' 'g' not as a single transition in
     *  the DFA.  Machine== o-'f'->o-'o'->o-'g'->o and has n+1 states
     *  for n characters.
     */
    public StateCluster build_StringLiteralAtom(GrammarAST stringLiteralAST) {
        if ( nfa.grammar.type==Grammar.LEXER ) {
			StringBuffer chars =
				Grammar.getUnescapedStringFromGrammarStringLiteral(stringLiteralAST.getText());
            NFAState first = newState();
            NFAState last = null;
            NFAState prev = first;
            for (int i=0; i<chars.length(); i++) {
                int c = chars.charAt(i);
                NFAState next = newState();
                transitionBetweenStates(prev, next, c);
                prev = last = next;
            }
            return  new StateCluster(first, last);
        }

        // a simple token reference in non-Lexers
        int tokenType = nfa.grammar.getTokenType(stringLiteralAST.getText());
		return build_Atom(tokenType, stringLiteralAST);
    }

    /** For reference to rule r, build
     *
     *  o-e->(r)  o
     *
     *  where (r) is the start of rule r and the trailing o is not linked
     *  to from rule ref state directly (it's done thru the transition(0)
     *  RuleClosureTransition.
     *
     *  If the rule r is just a list of tokens, it's block will be just
     *  a set on an edge o->o->o-set->o->o->o, could inline it rather than doing
     *  the rule reference, but i'm not doing this yet as I'm not sure
     *  it would help much in the NFA->DFA construction.
     *
     *  TODO add to codegen: collapse alt blks that are sets into single matchSet
     */
    public StateCluster build_RuleRef(Rule refDef, NFAState ruleStart) {
        //System.out.println("building ref to rule "+nfa.grammar.name+"."+refDef.name);
        NFAState left = newState();
        // left.setDescription("ref to "+ruleStart.getDescription());
        NFAState right = newState();
        // right.setDescription("NFAState following ref to "+ruleStart.getDescription());
        Transition e = new RuleClosureTransition(refDef,ruleStart,right);
        left.addTransition(e);
        StateCluster g = new StateCluster(left, right);
        return g;
    }

    /** From an empty alternative build StateCluster o-e->o */
    public StateCluster build_Epsilon() {
        NFAState left = newState();
        NFAState right = newState();
        transitionBetweenStates(left, right, Label.EPSILON);
        StateCluster g = new StateCluster(left, right);
        return g;
    }

	/** Build what amounts to an epsilon transition with a semantic
	 *  predicate action.  The pred is a pointer into the AST of
	 *  the SEMPRED token.
	 */
	public StateCluster build_SemanticPredicate(GrammarAST pred) {
		// don't count syn preds
		if ( !pred.getText().toUpperCase()
				.startsWith(Grammar.SYNPRED_RULE_PREFIX.toUpperCase()) )
		{
			nfa.grammar.numberOfSemanticPredicates++;
		}
		NFAState left = newState();
		NFAState right = newState();
		Transition e = new Transition(new PredicateLabel(pred), right);
		left.addTransition(e);
		StateCluster g = new StateCluster(left, right);
		return g;
	}

	/** Build what amounts to an epsilon transition with an action.
	 *  The action goes into NFA though it is ignored during analysis.
	 *  It slows things down a bit, but I must ignore predicates after
	 *  having seen an action (5-5-2008).
	 */
	public StateCluster build_Action(GrammarAST action) {
		NFAState left = newState();
		NFAState right = newState();
		Transition e = new Transition(new ActionLabel(action), right);
		left.addTransition(e);
		return new StateCluster(left, right);
	}

	/** add an EOF transition to any rule end NFAState that points to nothing
     *  (i.e., for all those rules not invoked by another rule).  These
     *  are start symbols then.
	 *
	 *  Return the number of grammar entry points; i.e., how many rules are
	 *  not invoked by another rule (they can only be invoked from outside).
	 *  These are the start rules.
     */
    public int build_EOFStates(List rules) {
		int numberUnInvokedRules = 0;
        for (Iterator iterator = rules.iterator(); iterator.hasNext();) {
			Rule r = (Rule) iterator.next();
			NFAState endNFAState = r.stopState;
            // Is this rule a start symbol?  (no follow links)
			if ( endNFAState.transition[0] ==null ) {
				// if so, then don't let algorithm fall off the end of
				// the rule, make it hit EOF/EOT.
				build_EOFState(endNFAState);
				// track how many rules have been invoked by another rule
				numberUnInvokedRules++;
			}
        }
		return numberUnInvokedRules;
    }

    /** set up an NFA NFAState that will yield eof tokens or,
     *  in the case of a lexer grammar, an EOT token when the conversion
     *  hits the end of a rule.
     */
    private void build_EOFState(NFAState endNFAState) {
		NFAState end = newState();
        int label = Label.EOF;
        if ( nfa.grammar.type==Grammar.LEXER ) {
            label = Label.EOT;
			end.setEOTTargetState(true);
        }
		/*
		System.out.println("build "+nfa.grammar.getTokenDisplayName(label)+
						   " loop on end of state "+endNFAState.getDescription()+
						   " to state "+end.stateNumber);
		*/
		Transition toEnd = new Transition(label, end);
		endNFAState.addTransition(toEnd);
	}

    /** From A B build A-e->B (that is, build an epsilon arc from right
     *  of A to left of B).
     *
     *  As a convenience, return B if A is null or return A if B is null.
     */
    public StateCluster build_AB(StateCluster A, StateCluster B) {
        if ( A==null ) {
            return B;
        }
        if ( B==null ) {
            return A;
        }
		transitionBetweenStates(A.right, B.left, Label.EPSILON);
		StateCluster g = new StateCluster(A.left, B.right);
        return g;
    }

	/** From a set ('a'|'b') build
     *
     *  o->o-'a'..'b'->o->o (last NFAState is blockEndNFAState pointed to by all alts)
	 */
	public StateCluster build_AlternativeBlockFromSet(StateCluster set) {
		if ( set==null ) {
			return null;
		}

		// single alt, no decision, just return only alt state cluster
		NFAState startOfAlt = newState(); // must have this no matter what
		transitionBetweenStates(startOfAlt, set.left, Label.EPSILON);

		return new StateCluster(startOfAlt,set.right);
	}

	/** From A|B|..|Z alternative block build
     *
     *  o->o-A->o->o (last NFAState is blockEndNFAState pointed to by all alts)
     *  |          ^
     *  o->o-B->o--|
     *  |          |
     *  ...        |
     *  |          |
     *  o->o-Z->o--|
     *
     *  So every alternative gets begin NFAState connected by epsilon
     *  and every alt right side points at a block end NFAState.  There is a
     *  new NFAState in the NFAState in the StateCluster for each alt plus one for the
     *  end NFAState.
     *
     *  Special case: only one alternative: don't make a block with alt
     *  begin/end.
     *
     *  Special case: if just a list of tokens/chars/sets, then collapse
     *  to a single edge'd o-set->o graph.
     *
     *  Set alt number (1..n) in the left-Transition NFAState.
     */
    public StateCluster build_AlternativeBlock(List alternativeStateClusters)
    {
        StateCluster result = null;
        if ( alternativeStateClusters==null || alternativeStateClusters.size()==0 ) {
            return null;
        }

		// single alt case
		if ( alternativeStateClusters.size()==1 ) {
			// single alt, no decision, just return only alt state cluster
			StateCluster g = (StateCluster)alternativeStateClusters.get(0);
			NFAState startOfAlt = newState(); // must have this no matter what
			transitionBetweenStates(startOfAlt, g.left, Label.EPSILON);

			//System.out.println("### opt saved start/stop end in (...)");
			return new StateCluster(startOfAlt,g.right);
		}

		// even if we can collapse for lookahead purposes, we will still
        // need to predict the alts of this subrule in case there are actions
        // etc...  This is the decision that is pointed to from the AST node
        // (always)
        NFAState prevAlternative = null; // tracks prev so we can link to next alt
        NFAState firstAlt = null;
        NFAState blockEndNFAState = newState();
        blockEndNFAState.setDescription("end block");
        int altNum = 1;
        for (Iterator iter = alternativeStateClusters.iterator(); iter.hasNext();) {
            StateCluster g = (StateCluster) iter.next();
            // add begin NFAState for this alt connected by epsilon
            NFAState left = newState();
            left.setDescription("alt "+altNum+" of ()");
			transitionBetweenStates(left, g.left, Label.EPSILON);
			transitionBetweenStates(g.right, blockEndNFAState, Label.EPSILON);
			// Are we the first alternative?
			if ( firstAlt==null ) {
				firstAlt = left; // track extreme left node of StateCluster
			}
			else {
				// if not first alternative, must link to this alt from previous
				transitionBetweenStates(prevAlternative, left, Label.EPSILON);
			}
			prevAlternative = left;
			altNum++;
		}

		// return StateCluster pointing representing entire block
		// Points to first alt NFAState on left, block end on right
		result = new StateCluster(firstAlt, blockEndNFAState);

		firstAlt.decisionStateType = NFAState.BLOCK_START;

		// set EOB markers for Jean
		firstAlt.endOfBlockStateNumber = blockEndNFAState.stateNumber;

		return result;
    }

    /** From (A)? build either:
     *
	 *  o--A->o
	 *  |     ^
	 *  o---->|
     *
     *  or, if A is a block, just add an empty alt to the end of the block
     */
    public StateCluster build_Aoptional(StateCluster A) {
        StateCluster g = null;
        int n = nfa.grammar.getNumberOfAltsForDecisionNFA(A.left);
        if ( n==1 ) {
            // no decision, just wrap in an optional path
			//NFAState decisionState = newState();
			NFAState decisionState = A.left; // resuse left edge
			decisionState.setDescription("only alt of ()? block");
			NFAState emptyAlt = newState();
            emptyAlt.setDescription("epsilon path of ()? block");
            NFAState blockEndNFAState = null;
			blockEndNFAState = newState();
			transitionBetweenStates(A.right, blockEndNFAState, Label.EPSILON);
			blockEndNFAState.setDescription("end ()? block");
            //transitionBetweenStates(decisionState, A.left, Label.EPSILON);
            transitionBetweenStates(decisionState, emptyAlt, Label.EPSILON);
            transitionBetweenStates(emptyAlt, blockEndNFAState, Label.EPSILON);

			// set EOB markers for Jean
			decisionState.endOfBlockStateNumber = blockEndNFAState.stateNumber;
			blockEndNFAState.decisionStateType = NFAState.RIGHT_EDGE_OF_BLOCK;

            g = new StateCluster(decisionState, blockEndNFAState);
        }
        else {
            // a decision block, add an empty alt
            NFAState lastRealAlt =
                    nfa.grammar.getNFAStateForAltOfDecision(A.left, n);
            NFAState emptyAlt = newState();
            emptyAlt.setDescription("epsilon path of ()? block");
            transitionBetweenStates(lastRealAlt, emptyAlt, Label.EPSILON);
            transitionBetweenStates(emptyAlt, A.right, Label.EPSILON);

			// set EOB markers for Jean (I think this is redundant here)
			A.left.endOfBlockStateNumber = A.right.stateNumber;
			A.right.decisionStateType = NFAState.RIGHT_EDGE_OF_BLOCK;

            g = A; // return same block, but now with optional last path
        }
		g.left.decisionStateType = NFAState.OPTIONAL_BLOCK_START;

        return g;
    }

    /** From (A)+ build
	 *
     *     |---|    (Transition 2 from A.right points at alt 1)
	 *     v   |    (follow of loop is Transition 1)
     *  o->o-A-o->o
     *
     *  Meaning that the last NFAState in A points back to A's left Transition NFAState
     *  and we add a new begin/end NFAState.  A can be single alternative or
     *  multiple.
	 *
	 *  During analysis we'll call the follow link (transition 1) alt n+1 for
	 *  an n-alt A block.
     */
    public StateCluster build_Aplus(StateCluster A) {
        NFAState left = newState();
        NFAState blockEndNFAState = newState();
		blockEndNFAState.decisionStateType = NFAState.RIGHT_EDGE_OF_BLOCK;

		// don't reuse A.right as loopback if it's right edge of another block
		if ( A.right.decisionStateType == NFAState.RIGHT_EDGE_OF_BLOCK ) {
			// nested A* so make another tail node to be the loop back
			// instead of the usual A.right which is the EOB for inner loop
			NFAState extraRightEdge = newState();
			transitionBetweenStates(A.right, extraRightEdge, Label.EPSILON);
			A.right = extraRightEdge;
		}

        transitionBetweenStates(A.right, blockEndNFAState, Label.EPSILON); // follow is Transition 1
		// turn A's block end into a loopback (acts like alt 2)
		transitionBetweenStates(A.right, A.left, Label.EPSILON); // loop back Transition 2
		transitionBetweenStates(left, A.left, Label.EPSILON);
		
		A.right.decisionStateType = NFAState.LOOPBACK;
		A.left.decisionStateType = NFAState.BLOCK_START;

		// set EOB markers for Jean
		A.left.endOfBlockStateNumber = A.right.stateNumber;

        StateCluster g = new StateCluster(left, blockEndNFAState);
        return g;
    }

    /** From (A)* build
     *
	 *     |---|
	 *     v   |
	 *  o->o-A-o--o (Transition 2 from block end points at alt 1; follow is Transition 1)
     *  |         ^
     *  o---------| (optional branch is 2nd alt of optional block containing A+)
     *
     *  Meaning that the last (end) NFAState in A points back to A's
     *  left side NFAState and we add 3 new NFAStates (the
     *  optional branch is built just like an optional subrule).
     *  See the Aplus() method for more on the loop back Transition.
	 *  The new node on right edge is set to RIGHT_EDGE_OF_CLOSURE so we
	 *  can detect nested (A*)* loops and insert an extra node.  Previously,
	 *  two blocks shared same EOB node.
     *
     *  There are 2 or 3 decision points in a A*.  If A is not a block (i.e.,
     *  it only has one alt), then there are two decisions: the optional bypass
     *  and then loopback.  If A is a block of alts, then there are three
     *  decisions: bypass, loopback, and A's decision point.
     *
     *  Note that the optional bypass must be outside the loop as (A|B)* is
     *  not the same thing as (A|B|)+.
     *
     *  This is an accurate NFA representation of the meaning of (A)*, but
     *  for generating code, I don't need a DFA for the optional branch by
     *  virtue of how I generate code.  The exit-loopback-branch decision
     *  is sufficient to let me make an appropriate enter, exit, loop
     *  determination.  See codegen.g
     */
    public StateCluster build_Astar(StateCluster A) {
		NFAState bypassDecisionState = newState();
		bypassDecisionState.setDescription("enter loop path of ()* block");
        NFAState optionalAlt = newState();
        optionalAlt.setDescription("epsilon path of ()* block");
        NFAState blockEndNFAState = newState();
		blockEndNFAState.decisionStateType = NFAState.RIGHT_EDGE_OF_BLOCK;

		// don't reuse A.right as loopback if it's right edge of another block
		if ( A.right.decisionStateType == NFAState.RIGHT_EDGE_OF_BLOCK ) {
			// nested A* so make another tail node to be the loop back
			// instead of the usual A.right which is the EOB for inner loop
			NFAState extraRightEdge = newState();
			transitionBetweenStates(A.right, extraRightEdge, Label.EPSILON);
			A.right = extraRightEdge;
		}

		// convert A's end block to loopback
		A.right.setDescription("()* loopback");
		// Transition 1 to actual block of stuff
        transitionBetweenStates(bypassDecisionState, A.left, Label.EPSILON);
        // Transition 2 optional to bypass
        transitionBetweenStates(bypassDecisionState, optionalAlt, Label.EPSILON);
		transitionBetweenStates(optionalAlt, blockEndNFAState, Label.EPSILON);
        // Transition 1 of end block exits
        transitionBetweenStates(A.right, blockEndNFAState, Label.EPSILON);
        // Transition 2 of end block loops
        transitionBetweenStates(A.right, A.left, Label.EPSILON);

		bypassDecisionState.decisionStateType = NFAState.BYPASS;
		A.left.decisionStateType = NFAState.BLOCK_START;
		A.right.decisionStateType = NFAState.LOOPBACK;

		// set EOB markers for Jean
		A.left.endOfBlockStateNumber = A.right.stateNumber;
		bypassDecisionState.endOfBlockStateNumber = blockEndNFAState.stateNumber;

        StateCluster g = new StateCluster(bypassDecisionState, blockEndNFAState);
        return g;
    }

    /** Build an NFA predictor for special rule called Tokens manually that
     *  predicts which token will succeed.  The refs to the rules are not
     *  RuleRefTransitions as I want DFA conversion to stop at the EOT
     *  transition on the end of each token, rather than return to Tokens rule.
     *  If I used normal build_alternativeBlock for this, the RuleRefTransitions
     *  would save return address when jumping away from Tokens rule.
     *
     *  All I do here is build n new states for n rules with an epsilon
     *  edge to the rule start states and then to the next state in the
     *  list:
     *
     *   o->(A)  (a state links to start of A and to next in list)
     *   |
     *   o->(B)
     *   |
     *   ...
     *   |
     *   o->(Z)
	 *
	 *  This is the NFA created for the artificial rule created in
	 *  Grammar.addArtificialMatchTokensRule().
	 *
	 *  11/28/2005: removed so we can use normal rule construction for Tokens.
    public NFAState build_ArtificialMatchTokensRuleNFA() {
        int altNum = 1;
        NFAState firstAlt = null; // the start state for the "rule"
        NFAState prevAlternative = null;
        Iterator iter = nfa.grammar.getRules().iterator();
		// TODO: add a single decision node/state for good description
        while (iter.hasNext()) {
			Rule r = (Rule) iter.next();
            String ruleName = r.name;
			String modifier = nfa.grammar.getRuleModifier(ruleName);
            if ( ruleName.equals(Grammar.ARTIFICIAL_TOKENS_RULENAME) ||
				 (modifier!=null &&
				  modifier.equals(Grammar.FRAGMENT_RULE_MODIFIER)) )
			{
                continue; // don't loop to yourself or do nontoken rules
            }
            NFAState ruleStartState = nfa.grammar.getRuleStartState(ruleName);
            NFAState left = newState();
            left.setDescription("alt "+altNum+" of artificial rule "+Grammar.ARTIFICIAL_TOKENS_RULENAME);
            transitionBetweenStates(left, ruleStartState, Label.EPSILON);
            // Are we the first alternative?
            if ( firstAlt==null ) {
                firstAlt = left; // track extreme top left node as rule start
            }
            else {
                // if not first alternative, must link to this alt from previous
                transitionBetweenStates(prevAlternative, left, Label.EPSILON);
            }
            prevAlternative = left;
            altNum++;
        }
		firstAlt.decisionStateType = NFAState.BLOCK_START;

        return firstAlt;
    }
	 */

    /** Build an atom with all possible values in its label */
    public StateCluster build_Wildcard(GrammarAST associatedAST) {
        NFAState left = newState();
        NFAState right = newState();
        left.associatedASTNode = associatedAST;
        right.associatedASTNode = associatedAST;
        Label label = new Label(nfa.grammar.getTokenTypes()); // char or tokens
        Transition e = new Transition(label,right);
        left.addTransition(e);
        StateCluster g = new StateCluster(left, right);
        return g;
    }

    /** Build a subrule matching ^(. .*) (any tree or node). Let's use
     *  (^(. .+) | .) to be safe.
     */
    public StateCluster build_WildcardTree(GrammarAST associatedAST) {
        StateCluster wildRoot = build_Wildcard(associatedAST);

        StateCluster down = build_Atom(Label.DOWN, associatedAST);
        wildRoot = build_AB(wildRoot,down); // hook in; . DOWN

        // make .+
        StateCluster wildChildren = build_Wildcard(associatedAST);
        wildChildren = build_Aplus(wildChildren);
        wildRoot = build_AB(wildRoot,wildChildren); // hook in; . DOWN .+

        StateCluster up = build_Atom(Label.UP, associatedAST);
        wildRoot = build_AB(wildRoot,up); // hook in; . DOWN .+ UP

        // make optional . alt
        StateCluster optionalNodeAlt = build_Wildcard(associatedAST);

        List alts = new ArrayList();
        alts.add(wildRoot);
        alts.add(optionalNodeAlt);
        StateCluster blk = build_AlternativeBlock(alts);

        return blk;
    }

    /** Given a collapsed block of alts (a set of atoms), pull out
     *  the set and return it.
     */
    protected IntSet getCollapsedBlockAsSet(State blk) {
        State s0 = blk;
        if ( s0!=null && s0.transition(0)!=null ) {
            State s1 = s0.transition(0).target;
            if ( s1!=null && s1.transition(0)!=null ) {
                Label label = s1.transition(0).label;
                if ( label.isSet() ) {
                    return label.getSet();
                }
            }
        }
        return null;
    }

	private void transitionBetweenStates(NFAState a, NFAState b, int label) {
		Transition e = new Transition(label,b);
		a.addTransition(e);
	}
}
