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
package org.antlr.analysis;

import org.antlr.tool.GrammarAST;
import org.antlr.tool.Rule;
import org.antlr.tool.ErrorManager;

/** A state within an NFA. At most 2 transitions emanate from any NFA state. */
public class NFAState extends State {
	// I need to distinguish between NFA decision states for (...)* and (...)+
	// during NFA interpretation.
	public static final int LOOPBACK = 1;
	public static final int BLOCK_START = 2;
	public static final int OPTIONAL_BLOCK_START = 3;
	public static final int BYPASS = 4;
	public static final int RIGHT_EDGE_OF_BLOCK = 5;

	public static final int MAX_TRANSITIONS = 2;

	/** How many transitions; 0, 1, or 2 transitions */
	int numTransitions = 0;
	public Transition[] transition = new Transition[MAX_TRANSITIONS];

	/** For o-A->o type NFA tranitions, record the label that leads to this
	 *  state.  Useful for creating rich error messages when we find
	 *  insufficiently (with preds) covered states.
	 */
	public Label incidentEdgeLabel;

	/** Which NFA are we in? */
	public NFA nfa = null;

	/** What's its decision number from 1..n? */
	protected int decisionNumber = 0;

	/** Subrules (...)* and (...)+ have more than one decision point in
	 *  the NFA created for them.  They both have a loop-exit-or-stay-in
	 *  decision node (the loop back node).  They both have a normal
	 *  alternative block decision node at the left edge.  The (...)* is
	 *  worse as it even has a bypass decision (2 alts: stay in or bypass)
	 *  node at the extreme left edge.  This is not how they get generated
	 *  in code as a while-loop or whatever deals nicely with either.  For
	 *  error messages (where I need to print the nondeterministic alts)
	 *  and for interpretation, I need to use the single DFA that is created
	 *  (for efficiency) but interpret the results differently depending
	 *  on which of the 2 or 3 decision states uses the DFA.  For example,
	 *  the DFA will always report alt n+1 as the exit branch for n real
	 *  alts, so I need to translate that depending on the decision state.
	 *
	 *  If decisionNumber>0 then this var tells you what kind of decision
	 *  state it is.
	 */
	public int decisionStateType;

	/** What rule do we live in? */
	public Rule enclosingRule;

	/** During debugging and for nondeterminism warnings, it's useful
	 *  to know what relationship this node has to the original grammar.
	 *  For example, "start of alt 1 of rule a".
	 */
	protected String description;

	/** Associate this NFAState with the corresponding GrammarAST node
	 *  from which this node was created.  This is useful not only for
	 *  associating the eventual lookahead DFA with the associated
	 *  Grammar position, but also for providing users with
	 *  nondeterminism warnings.  Mainly used by decision states to
	 *  report line:col info.  Could also be used to track line:col
	 *  for elements such as token refs.
	 */
	public GrammarAST associatedASTNode;

	/** Is this state the sole target of an EOT transition? */
	protected boolean EOTTargetState = false;

	/** Jean Bovet needs in the GUI to know which state pairs correspond
	 *  to the start/stop of a block.
	  */
	public int endOfBlockStateNumber = State.INVALID_STATE_NUMBER;

	public NFAState(NFA nfa) {
		this.nfa = nfa;
	}

	public int getNumberOfTransitions() {
		return numTransitions;
	}

	public void addTransition(Transition e) {
		if ( e==null ) {
			throw new IllegalArgumentException("You can't add a null transition");			
		}
		if ( numTransitions>transition.length ) {
			throw new IllegalArgumentException("You can only have "+transition.length+" transitions");
		}
		if ( e!=null ) {
			transition[numTransitions] = e;
			numTransitions++;
			// Set the "back pointer" of the target state so that it
			// knows about the label of the incoming edge.
			Label label = e.label;
			if ( label.isAtom() || label.isSet() ) {
				if ( ((NFAState)e.target).incidentEdgeLabel!=null ) {
					ErrorManager.internalError("Clobbered incident edge");
				}
				((NFAState)e.target).incidentEdgeLabel = e.label;
			}
		}
	}

	/** Used during optimization to reset a state to have the (single)
	 *  transition another state has.
	 */
	public void setTransition0(Transition e) {
		if ( e==null ) {
			throw new IllegalArgumentException("You can't use a solitary null transition");
		}
		transition[0] = e;
		transition[1] = null;
		numTransitions = 1;
	}

	public Transition transition(int i) {
		return transition[i];
	}

	/** The DFA decision for this NFA decision state always has
	 *  an exit path for loops as n+1 for n alts in the loop.
	 *  That is really useful for displaying nondeterministic alts
	 *  and so on, but for walking the NFA to get a sequence of edge
	 *  labels or for actually parsing, we need to get the real alt
	 *  number.  The real alt number for exiting a loop is always 1
	 *  as transition 0 points at the exit branch (we compute DFAs
	 *  always for loops at the loopback state).
	 *
	 *  For walking/parsing the loopback state:
	 * 		1 2 3 display alt (for human consumption)
	 * 		2 3 1 walk alt
	 *
	 *  For walking the block start:
	 * 		1 2 3 display alt
	 * 		1 2 3
	 *
	 *  For walking the bypass state of a (...)* loop:
	 * 		1 2 3 display alt
	 * 		1 1 2 all block alts map to entering loop exit means take bypass
	 *
	 *  Non loop EBNF do not need to be translated; they are ignored by
	 *  this method as decisionStateType==0.
	 *
	 *  Return same alt if we can't translate.
	 */
	public int translateDisplayAltToWalkAlt(int displayAlt) {
		NFAState nfaStart = this;
		if ( decisionNumber==0 || decisionStateType==0 ) {
			return displayAlt;
		}
		int walkAlt = 0;
		// find the NFA loopback state associated with this DFA
		// and count number of alts (all alt numbers are computed
		// based upon the loopback's NFA state.
		/*
		DFA dfa = nfa.grammar.getLookaheadDFA(decisionNumber);
		if ( dfa==null ) {
			ErrorManager.internalError("can't get DFA for decision "+decisionNumber);
		}
		*/
		int nAlts = nfa.grammar.getNumberOfAltsForDecisionNFA(nfaStart);
		switch ( nfaStart.decisionStateType ) {
			case LOOPBACK :
				walkAlt = displayAlt % nAlts + 1; // rotate right mod 1..3
				break;
			case BLOCK_START :
			case OPTIONAL_BLOCK_START :
				walkAlt = displayAlt; // identity transformation
				break;
			case BYPASS :
				if ( displayAlt == nAlts ) {
					walkAlt = 2; // bypass
				}
				else {
					walkAlt = 1; // any non exit branch alt predicts entering
				}
				break;
		}
		return walkAlt;
	}

	// Setter/Getters

	/** What AST node is associated with this NFAState?  When you
	 *  set the AST node, I set the node to point back to this NFA state.
	 */
	public void setDecisionASTNode(GrammarAST decisionASTNode) {
		decisionASTNode.setNFAStartState(this);
		this.associatedASTNode = decisionASTNode;
	}

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public int getDecisionNumber() {
		return decisionNumber;
	}

	public void setDecisionNumber(int decisionNumber) {
		this.decisionNumber = decisionNumber;
	}

	public boolean isEOTTargetState() {
		return EOTTargetState;
	}

	public void setEOTTargetState(boolean eot) {
		EOTTargetState = eot;
	}

	public boolean isDecisionState() {
		return decisionStateType>0;
	}

	public String toString() {
		return String.valueOf(stateNumber);
	}

}

