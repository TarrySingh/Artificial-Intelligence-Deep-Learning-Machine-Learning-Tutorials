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

import org.antlr.misc.Utils;

/** An NFA state, predicted alt, and syntactic/semantic context.
 *  The syntactic context is a pointer into the rule invocation
 *  chain used to arrive at the state.  The semantic context is
 *  the unordered set semantic predicates encountered before reaching
 *  an NFA state.
 */
public class NFAConfiguration {
    /** The NFA state associated with this configuration */
    public int state;

    /** What alt is predicted by this configuration */
    public int alt;

    /** What is the stack of rule invocations that got us to state? */
    public NFAContext context;

    /** The set of semantic predicates associated with this NFA
     *  configuration.  The predicates were found on the way to
     *  the associated NFA state in this syntactic context.
     *  Set<AST>: track nodes in grammar containing the predicate
     *  for error messages and such (nice to know where the predicate
     *  came from in case of duplicates etc...).  By using a set,
     *  the equals() method will correctly show {pred1,pred2} as equals()
     *  to {pred2,pred1}.
     */
    public SemanticContext semanticContext = SemanticContext.EMPTY_SEMANTIC_CONTEXT;

    /** Indicate that this configuration has been resolved and no further
     *  DFA processing should occur with it.  Essentially, this is used
     *  as an "ignore" bit so that upon a set of nondeterministic configurations
     *  such as (s|2) and (s|3), I can set (s|3) to resolved=true (and any
     *  other configuration associated with alt 3).
     */
    protected boolean resolved;

    /** This bit is used to indicate a semantic predicate will be
     *  used to resolve the conflict.  Method
     *  DFA.findNewDFAStatesAndAddDFATransitions will add edges for
     *  the predicates after it performs the reach operation.  The
     *  nondeterminism resolver sets this when it finds a set of
     *  nondeterministic configurations (as it does for "resolved" field)
     *  that have enough predicates to resolve the conflit.
     */
    protected boolean resolveWithPredicate;

    /** Lots of NFA states have only epsilon edges (1 or 2).  We can
     *  safely consider only n>0 during closure.
     */
    protected int numberEpsilonTransitionsEmanatingFromState;

    /** Indicates that the NFA state associated with this configuration
     *  has exactly one transition and it's an atom (not epsilon etc...).
     */
    protected boolean singleAtomTransitionEmanating;

	//protected boolean addedDuringClosure = true;

	public NFAConfiguration(int state,
                            int alt,
                            NFAContext context,
                            SemanticContext semanticContext)
    {
        this.state = state;
        this.alt = alt;
        this.context = context;
        this.semanticContext = semanticContext;
    }

    /** An NFA configuration is equal to another if both have
     *  the same state, the predict the same alternative, and
     *  syntactic/semantic contexts are the same.  I don't think
     *  the state|alt|ctx could be the same and have two different
     *  semantic contexts, but might as well define equals to be
     *  everything.
     */
    public boolean equals(Object o) {
		if ( o==null ) {
			return false;
		}
        NFAConfiguration other = (NFAConfiguration)o;
        return this.state==other.state &&
               this.alt==other.alt &&
               this.context.equals(other.context)&&
               this.semanticContext.equals(other.semanticContext);
    }

    public int hashCode() {
        int h = state + alt + context.hashCode();
        return h;
    }

	public String toString() {
		return toString(true);
	}

	public String toString(boolean showAlt) {
		StringBuffer buf = new StringBuffer();
		buf.append(state);
		if ( showAlt ) {
			buf.append("|");
			buf.append(alt);
		}
		if ( context.parent!=null ) {
            buf.append("|");
            buf.append(context);
        }
        if ( semanticContext!=null &&
             semanticContext!=SemanticContext.EMPTY_SEMANTIC_CONTEXT ) {
            buf.append("|");
			String escQuote = Utils.replace(semanticContext.toString(), "\"", "\\\"");
			buf.append(escQuote);
        }
        if ( resolved ) {
            buf.append("|resolved");
        }
		if ( resolveWithPredicate ) {
			buf.append("|resolveWithPredicate");
		}
		return buf.toString();
    }
}
