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

/** A generic transition between any two state machine states.  It defines
 *  some special labels that indicate things like epsilon transitions and
 *  that the label is actually a set of labels or a semantic predicate.
 *  This is a one way link.  It emanates from a state (usually via a list of
 *  transitions) and has a label/target pair.  I have abstracted the notion
 *  of a Label to handle the various kinds of things it can be.
 */
public class Transition implements Comparable {
    /** What label must be consumed to transition to target */
    public Label label;

    /** The target of this transition */
    public State target;

    public Transition(Label label, State target) {
        this.label = label;
        this.target = target;
    }

    public Transition(int label, State target) {
        this.label = new Label(label);
        this.target = target;
    }

	public boolean isEpsilon() {
		return label.isEpsilon();
	}

	public boolean isAction() {
		return label.isAction();
	}

    public boolean isSemanticPredicate() {
        return label.isSemanticPredicate();
    }

    public int hashCode() {
        return label.hashCode() + target.stateNumber;
    }

    public boolean equals(Object o) {
        Transition other = (Transition)o;
        return this.label.equals(other.label) &&
               this.target.equals(other.target);
    }

    public int compareTo(Object o) {
        Transition other = (Transition)o;
        return this.label.compareTo(other.label);
    }

    public String toString() {
        return label+"->"+target.stateNumber;
    }
}
