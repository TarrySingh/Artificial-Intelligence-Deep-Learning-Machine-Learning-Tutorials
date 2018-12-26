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

import org.antlr.tool.Grammar;
import org.antlr.misc.Utils;

import java.util.HashSet;
import java.util.Set;

/** A module to perform optimizations on DFAs.
 *
 *  I could more easily (and more quickly) do some optimizations (such as
 *  PRUNE_EBNF_EXIT_BRANCHES) during DFA construction, but then it
 *  messes up the determinism checking.  For example, it looks like
 *  loop exit branches are unreachable if you prune exit branches
 *  during DFA construction and before determinism checks.
 *
 *  In general, ANTLR's NFA->DFA->codegen pipeline seems very robust
 *  to me which I attribute to a uniform and consistent set of data
 *  structures.  Regardless of what I want to "say"/implement, I do so
 *  within the confines of, for example, a DFA.  The code generator
 *  can then just generate code--it doesn't have to do much thinking.
 *  Putting optimizations in the code gen code really starts to make
 *  it a spagetti factory (uh oh, now I'm hungry!).  The pipeline is
 *  very testable; each stage has well defined input/output pairs.
 *
 *  ### Optimization: PRUNE_EBNF_EXIT_BRANCHES
 *
 *  There is no need to test EBNF block exit branches.  Not only is it
 *  an unneeded computation, but counter-intuitively, you actually get
 *  better errors. You can report an error at the missing or extra
 *  token rather than as soon as you've figured out you will fail.
 *
 *  Imagine optional block "( DOT CLASS )? SEMI".  ANTLR generates:
 *
 *  int alt=0;
 *  if ( input.LA(1)==DOT ) {
 *      alt=1;
 *  }
 *  else if ( input.LA(1)==SEMI ) {
 *      alt=2;
 *  }
 *
 *  Clearly, since Parser.match() will ultimately find the error, we
 *  do not want to report an error nor do we want to bother testing
 *  lookahead against what follows the (...)?  We want to generate
 *  simply "should I enter the subrule?":
 *
 *  int alt=2;
 *  if ( input.LA(1)==DOT ) {
 *      alt=1;
 *  }
 *
 *  NOTE 1. Greedy loops cannot be optimized in this way.  For example,
 *  "(greedy=false:'x'|.)* '\n'".  You specifically need the exit branch
 *  to tell you when to terminate the loop as the same input actually
 *  predicts one of the alts (i.e., staying in the loop).
 *
 *  NOTE 2.  I do not optimize cyclic DFAs at the moment as it doesn't
 *  seem to work. ;)  I'll have to investigate later to see what work I
 *  can do on cyclic DFAs to make them have fewer edges.  Might have
 *  something to do with the EOT token.
 *
 *  ### PRUNE_SUPERFLUOUS_EOT_EDGES
 *
 *  When a token is a subset of another such as the following rules, ANTLR
 *  quietly assumes the first token to resolve the ambiguity.
 *
 *  EQ			: '=' ;
 *  ASSIGNOP	: '=' | '+=' ;
 *
 *  It can yield states that have only a single edge on EOT to an accept
 *  state.  This is a waste and messes up my code generation. ;)  If
 *  Tokens rule DFA goes
 *
 * 		s0 -'='-> s3 -EOT-> s5 (accept)
 *
 *  then s5 should be pruned and s3 should be made an accept.  Do NOT do this
 *  for keyword versus ID as the state with EOT edge emanating from it will
 *  also have another edge.
 *
 *  ### Optimization: COLLAPSE_ALL_INCIDENT_EDGES
 *
 *  Done during DFA construction.  See method addTransition() in
 *  NFAToDFAConverter.
 *
 *  ### Optimization: MERGE_STOP_STATES
 *
 *  Done during DFA construction.  See addDFAState() in NFAToDFAConverter.
 */
public class DFAOptimizer {
	public static boolean PRUNE_EBNF_EXIT_BRANCHES = true;
	public static boolean PRUNE_TOKENS_RULE_SUPERFLUOUS_EOT_EDGES = true;
	public static boolean COLLAPSE_ALL_PARALLEL_EDGES = true;
	public static boolean MERGE_STOP_STATES = true;

	/** Used by DFA state machine generator to avoid infinite recursion
	 *  resulting from cycles int the DFA.  This is a set of int state #s.
	 *  This is a side-effect of calling optimize; can't clear after use
	 *  because code gen needs it.
	 */
	protected Set visited = new HashSet();

    protected Grammar grammar;

    public DFAOptimizer(Grammar grammar) {
		this.grammar = grammar;
    }

	public void optimize() {
		// optimize each DFA in this grammar
		for (int decisionNumber=1;
			 decisionNumber<=grammar.getNumberOfDecisions();
			 decisionNumber++)
		{
			DFA dfa = grammar.getLookaheadDFA(decisionNumber);
			optimize(dfa);
		}
	}

	protected void optimize(DFA dfa) {
		if ( dfa==null ) {
			return; // nothing to do
		}
		/*
		System.out.println("Optimize DFA "+dfa.decisionNFAStartState.decisionNumber+
						   " num states="+dfa.getNumberOfStates());
		*/
		//long start = System.currentTimeMillis();
		if ( PRUNE_EBNF_EXIT_BRANCHES && dfa.canInlineDecision() ) {
			visited.clear();
			int decisionType =
				dfa.getNFADecisionStartState().decisionStateType;
			if ( dfa.isGreedy() &&
				 (decisionType==NFAState.OPTIONAL_BLOCK_START ||
				 decisionType==NFAState.LOOPBACK) )
			{
				optimizeExitBranches(dfa.startState);
			}
		}
		// If the Tokens rule has syntactically ambiguous rules, try to prune
		if ( PRUNE_TOKENS_RULE_SUPERFLUOUS_EOT_EDGES &&
			 dfa.isTokensRuleDecision() &&
			 dfa.probe.stateToSyntacticallyAmbiguousTokensRuleAltsMap.size()>0 )
		{
			visited.clear();
			optimizeEOTBranches(dfa.startState);
		}

		/* ack...code gen needs this, cannot optimize
		visited.clear();
		unlinkUnneededStateData(dfa.startState);
		*/
		//long stop = System.currentTimeMillis();
		//System.out.println("minimized in "+(int)(stop-start)+" ms");
    }

	protected void optimizeExitBranches(DFAState d) {
		Integer sI = Utils.integer(d.stateNumber);
		if ( visited.contains(sI) ) {
			return; // already visited
		}
		visited.add(sI);
		int nAlts = d.dfa.getNumberOfAlts();
		for (int i = 0; i < d.getNumberOfTransitions(); i++) {
			Transition edge = (Transition) d.transition(i);
			DFAState edgeTarget = ((DFAState)edge.target);
			/*
			System.out.println(d.stateNumber+"-"+
							   edge.label.toString(d.dfa.nfa.grammar)+"->"+
							   edgeTarget.stateNumber);
			*/
			// if target is an accept state and that alt is the exit alt
			if ( edgeTarget.isAcceptState() &&
				edgeTarget.getUniquelyPredictedAlt()==nAlts)
			{
				/*
				System.out.println("ignoring transition "+i+" to max alt "+
					d.dfa.getNumberOfAlts());
				*/
				d.removeTransition(i);
				i--; // back up one so that i++ of loop iteration stays within bounds
			}
			optimizeExitBranches(edgeTarget);
		}
	}

	protected void optimizeEOTBranches(DFAState d) {
		Integer sI = Utils.integer(d.stateNumber);
		if ( visited.contains(sI) ) {
			return; // already visited
		}
		visited.add(sI);
		for (int i = 0; i < d.getNumberOfTransitions(); i++) {
			Transition edge = (Transition) d.transition(i);
			DFAState edgeTarget = ((DFAState)edge.target);
			/*
			System.out.println(d.stateNumber+"-"+
							   edge.label.toString(d.dfa.nfa.grammar)+"->"+
							   edgeTarget.stateNumber);
			*/
			// if only one edge coming out, it is EOT, and target is accept prune
			if ( PRUNE_TOKENS_RULE_SUPERFLUOUS_EOT_EDGES &&
				edgeTarget.isAcceptState() &&
				d.getNumberOfTransitions()==1 &&
				edge.label.isAtom() &&
				edge.label.getAtom()==Label.EOT )
			{
				//System.out.println("state "+d+" can be pruned");
				// remove the superfluous EOT edge
				d.removeTransition(i);
				d.setAcceptState(true); // make it an accept state
				// force it to uniquely predict the originally predicted state
				d.cachedUniquelyPredicatedAlt =
					edgeTarget.getUniquelyPredictedAlt();
				i--; // back up one so that i++ of loop iteration stays within bounds
			}
			optimizeEOTBranches(edgeTarget);
		}
	}

	/** Walk DFA states, unlinking the nfa configs and whatever else I
	 *  can to reduce memory footprint.
	protected void unlinkUnneededStateData(DFAState d) {
		Integer sI = Utils.integer(d.stateNumber);
		if ( visited.contains(sI) ) {
			return; // already visited
		}
		visited.add(sI);
		d.nfaConfigurations = null;
		for (int i = 0; i < d.getNumberOfTransitions(); i++) {
			Transition edge = (Transition) d.transition(i);
			DFAState edgeTarget = ((DFAState)edge.target);
			unlinkUnneededStateData(edgeTarget);
		}
	}
	 */

}
