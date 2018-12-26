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

import java.util.*;

/** An aspect of FA (finite automata) that knows how to dump them to serialized
 *  strings.
 */
public class FASerializer {
    /** To prevent infinite recursion when walking state machines, record
     *  which states we've visited.  Make a new set every time you start
     *  walking in case you reuse this object.  Multiple threads will trash
     *  this shared variable.  Use a different FASerializer per thread.
     */
    protected Set markedStates;

    /** Each state we walk will get a new state number for serialization
     *  purposes.  This is the variable that tracks state numbers.
     */
    protected int stateCounter = 0;

    /** Rather than add a new instance variable to NFA and DFA just for
     *  serializing machines, map old state numbers to new state numbers
     *  by a State object -> Integer new state number HashMap.
     */
    protected Map stateNumberTranslator;

    protected Grammar grammar;

    /** This aspect is associated with a grammar; used to get token names */
    public FASerializer(Grammar grammar) {
        this.grammar = grammar;
    }

	public String serialize(State s) {
		if ( s==null ) {
			return "<no automaton>";
		}
		return serialize(s, true);
	}

	/** Return a string representation of a state machine.  Two identical
     *  NFAs or DFAs will have identical serialized representations.  The
     *  state numbers inside the state are not used; instead, a new number
     *  is computed and because the serialization will walk the two
     *  machines using the same specific algorithm, then the state numbers
     *  will be identical.  Accept states are distinguished from regular
     *  states.
     */
    public String serialize(State s, boolean renumber) {
        markedStates = new HashSet();
        stateCounter = 0;
		if ( renumber ) {
			stateNumberTranslator = new HashMap();
        	walkFANormalizingStateNumbers(s);
		}
		List lines = new ArrayList();
        if ( s.getNumberOfTransitions()>0 ) {
			walkSerializingFA(lines, s);
		}
		else {
			// special case: s0 is an accept
			String s0 = getStateString(0, s);
			lines.add(s0+"\n");
		}
        StringBuffer buf = new StringBuffer(0);
        // sort lines to normalize; makes states come out ordered
        // and then ordered by edge labels then by target state number :)
        Collections.sort(lines);
        for (int i = 0; i < lines.size(); i++) {
            String line = (String) lines.get(i);
            buf.append(line);
        }
        return buf.toString();
    }

    /** In stateNumberTranslator, get a map from State to new, normalized
     *  state number.  Used by walkSerializingFA to make sure any two
     *  identical state machines will serialize the same way.
     */
    protected void walkFANormalizingStateNumbers(State s) {
		if ( s==null ) {
			ErrorManager.internalError("null state s");
			return;
		}
        if ( stateNumberTranslator.get(s)!=null ) {
            return; // already did this state
        }
        // assign a new state number for this node if there isn't one
        stateNumberTranslator.put(s, Utils.integer(stateCounter));
        stateCounter++;

        // visit nodes pointed to by each transition;
        for (int i = 0; i < s.getNumberOfTransitions(); i++) {
            Transition edge = (Transition) s.transition(i);
            walkFANormalizingStateNumbers(edge.target); // keep walkin'
            // if this transition is a rule reference, the node "following" this state
            // will not be found and appear to be not in graph.  Must explicitly jump
            // to it, but don't "draw" an edge.
            if ( edge instanceof RuleClosureTransition ) {
				walkFANormalizingStateNumbers(((RuleClosureTransition) edge).followState);
            }
        }
    }

    protected void walkSerializingFA(List lines, State s) {
        if ( markedStates.contains(s) ) {
            return; // already visited this node
        }

        markedStates.add(s); // mark this node as completed.

		int normalizedStateNumber = s.stateNumber;
		if ( stateNumberTranslator!=null ) {
	        Integer normalizedStateNumberI = (Integer)stateNumberTranslator.get(s);
			normalizedStateNumber = normalizedStateNumberI.intValue();
		}

		String stateStr = getStateString(normalizedStateNumber, s);

        // depth first walk each transition, printing its edge first
        for (int i = 0; i < s.getNumberOfTransitions(); i++) {
            Transition edge = (Transition) s.transition(i);
            StringBuffer buf = new StringBuffer();
            buf.append(stateStr);
			if ( edge.isAction() ) {
				buf.append("-{}->");
			}
			else if ( edge.isEpsilon() ) {
				buf.append("->");
			}
			else if ( edge.isSemanticPredicate() ) {
				buf.append("-{"+edge.label.getSemanticContext()+"}?->");
			}
			else {
				String predsStr = "";
				if ( edge.target instanceof DFAState ) {
					// look for gated predicates; don't add gated to simple sempred edges
					SemanticContext preds =
						((DFAState)edge.target).getGatedPredicatesInNFAConfigurations();
					if ( preds!=null ) {
						predsStr = "&&{"+
							preds.genExpr(grammar.generator,
									   	  grammar.generator.getTemplates(), null).toString()
							+"}?";
					}
				}
				buf.append("-"+edge.label.toString(grammar)+predsStr+"->");
			}

			int normalizedTargetStateNumber = edge.target.stateNumber;
			if ( stateNumberTranslator!=null ) {
				Integer normalizedTargetStateNumberI =
					(Integer)stateNumberTranslator.get(edge.target);
				normalizedTargetStateNumber = normalizedTargetStateNumberI.intValue();
			}
			buf.append(getStateString(normalizedTargetStateNumber, edge.target));
            buf.append("\n");
            lines.add(buf.toString());

            // walk this transition
            walkSerializingFA(lines, edge.target);

            // if this transition is a rule reference, the node "following" this state
            // will not be found and appear to be not in graph.  Must explicitly jump
            // to it, but don't "draw" an edge.
            if ( edge instanceof RuleClosureTransition ) {
				walkSerializingFA(lines, ((RuleClosureTransition) edge).followState);
            }
        }

    }

    private String getStateString(int n, State s) {
        String stateStr = ".s"+n;
        if ( s.isAcceptState() ) {
            if ( s instanceof DFAState ) {
                stateStr = ":s"+n+"=>"+((DFAState)s).getUniquelyPredictedAlt();
            }
            else {
                stateStr = ":s"+n;
            }
        }
        return stateStr;
    }


}
