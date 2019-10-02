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
import org.antlr.tool.GrammarAST;
import org.antlr.misc.IntervalSet;
import org.antlr.misc.IntSet;

/** A state machine transition label.  A label can be either a simple
 *  label such as a token or character.  A label can be a set of char or
 *  tokens.  It can be an epsilon transition.  It can be a semantic predicate
 *  (which assumes an epsilon transition) or a tree of predicates (in a DFA).
 */
public class Label implements Comparable, Cloneable {
    public static final int INVALID = -7;

	public static final int ACTION = -6;
	
	public static final int EPSILON = -5;

    public static final String EPSILON_STR = "<EPSILON>";

    /** label is a semantic predicate; implies label is epsilon also */
    public static final int SEMPRED = -4;

    /** label is a set of tokens or char */
    public static final int SET = -3;

    /** End of Token is like EOF for lexer rules.  It implies that no more
     *  characters are available and that NFA conversion should terminate
     *  for this path.  For example
     *
     *  A : 'a' 'b' | 'a' ;
     *
     *  yields a DFA predictor:
     *
     *  o-a->o-b->1   predict alt 1
     *       |
     *       |-EOT->o predict alt 2
     *
     *  To generate code for EOT, treat it as the "default" path, which
     *  implies there is no way to mismatch a char for the state from
     *  which the EOT emanates.
     */
    public static final int EOT = -2;

    public static final int EOF = -1;

	/** We have labels like EPSILON that are below 0; it's hard to
	 *  store them in an array with negative index so use this
	 *  constant as an index shift when accessing arrays based upon
	 *  token type.  If real token type is i, then array index would be
	 *  NUM_FAUX_LABELS + i.
	 */
	public static final int NUM_FAUX_LABELS = -INVALID;

    /** Anything at this value or larger can be considered a simple atom int
     *  for easy comparison during analysis only; faux labels are not used
	 *  during parse time for real token types or char values.
     */
    public static final int MIN_ATOM_VALUE = EOT;

    // TODO: is 0 a valid unicode char? max is FFFF -1, right?
    public static final int MIN_CHAR_VALUE = '\u0000';
    public static final int MAX_CHAR_VALUE = '\uFFFF';

	/** End of rule token type; imaginary token type used only for
	 *  local, partial FOLLOW sets to indicate that the local FOLLOW
	 *  hit the end of rule.  During error recovery, the local FOLLOW
	 *  of a token reference may go beyond the end of the rule and have
	 *  to use FOLLOW(rule).  I have to just shift the token types to 2..n
	 *  rather than 1..n to accommodate this imaginary token in my bitsets.
	 *  If I didn't use a bitset implementation for runtime sets, I wouldn't
	 *  need this.  EOF is another candidate for a run time token type for
	 *  parsers.  Follow sets are not computed for lexers so we do not have
	 *  this issue.
	 */
	public static final int EOR_TOKEN_TYPE =
		org.antlr.runtime.Token.EOR_TOKEN_TYPE;

	public static final int DOWN = org.antlr.runtime.Token.DOWN;
	public static final int UP = org.antlr.runtime.Token.UP;

    /** tokens and char range overlap; tokens are MIN_TOKEN_TYPE..n */
	public static final int MIN_TOKEN_TYPE =
		org.antlr.runtime.Token.MIN_TOKEN_TYPE;

    /** The wildcard '.' char atom implies all valid characters==UNICODE */
    //public static final IntSet ALLCHAR = IntervalSet.of(MIN_CHAR_VALUE,MAX_CHAR_VALUE);

    /** The token type or character value; or, signifies special label. */
    protected int label;

    /** A set of token types or character codes if label==SET */
	// TODO: try IntervalSet for everything
    protected IntSet labelSet;

    public Label(int label) {
        this.label = label;
    }

    /** Make a set label */
    public Label(IntSet labelSet) {
		if ( labelSet==null ) {
			this.label = SET;
			this.labelSet = IntervalSet.of(INVALID);
			return;
		}
		int singleAtom = labelSet.getSingleElement();
        if ( singleAtom!=INVALID ) {
            // convert back to a single atomic element if |labelSet|==1
            label = singleAtom;
            return;
        }
        this.label = SET;
        this.labelSet = labelSet;
    }

	public Object clone() {
		Label l;
		try {
			l = (Label)super.clone();
			l.label = this.label;
            l.labelSet = new IntervalSet();
			l.labelSet.addAll(this.labelSet);
		}
		catch (CloneNotSupportedException e) {
			throw new InternalError();
		}
		return l;
	}

	public void add(Label a) {
		if ( isAtom() ) {
			labelSet = IntervalSet.of(label);
			label=SET;
			if ( a.isAtom() ) {
				labelSet.add(a.getAtom());
			}
			else if ( a.isSet() ) {
				labelSet.addAll(a.getSet());
			}
			else {
				throw new IllegalStateException("can't add element to Label of type "+label);
			}
			return;
		}
		if ( isSet() ) {
			if ( a.isAtom() ) {
				labelSet.add(a.getAtom());
			}
			else if ( a.isSet() ) {
				labelSet.addAll(a.getSet());
			}
			else {
				throw new IllegalStateException("can't add element to Label of type "+label);
			}
			return;
		}
		throw new IllegalStateException("can't add element to Label of type "+label);
	}

    public boolean isAtom() {
        return label>=MIN_ATOM_VALUE;
    }

    public boolean isEpsilon() {
        return label==EPSILON;
    }

	public boolean isSemanticPredicate() {
		return false;
	}

	public boolean isAction() {
		return false;
	}

    public boolean isSet() {
        return label==SET;
    }

    /** return the single atom label or INVALID if not a single atom */
    public int getAtom() {
        if ( isAtom() ) {
            return label;
        }
        return INVALID;
    }

    public IntSet getSet() {
        if ( label!=SET ) {
            // convert single element to a set if they ask for it.
            return IntervalSet.of(label);
        }
        return labelSet;
    }

    public void setSet(IntSet set) {
        label=SET;
        labelSet = set;
    }

    public SemanticContext getSemanticContext() {
        return null;
    }

	public boolean matches(int atom) {
		if ( label==atom ) {
			return true; // handle the single atom case efficiently
		}
		if ( isSet() ) {
			return labelSet.member(atom);
		}
		return false;
	}

	public boolean matches(IntSet set) {
		if ( isAtom() ) {
			return set.member(getAtom());
		}
		if ( isSet() ) {
			// matches if intersection non-nil
			return !getSet().and(set).isNil();
		}
		return false;
	}


	public boolean matches(Label other) {
		if ( other.isSet() ) {
			return matches(other.getSet());
		}
		if ( other.isAtom() ) {
			return matches(other.getAtom());
		}
		return false;
	}

    public int hashCode() {
        if (label==SET) {
            return labelSet.hashCode();
		}
		else {
			return label;
		}
	}

	// TODO: do we care about comparing set {A} with atom A? Doesn't now.
	public boolean equals(Object o) {
		if ( o==null ) {
			return false;
		}
		if ( this == o ) {
			return true; // equals if same object
		}
		// labels must be the same even if epsilon or set or sempred etc...
        if ( label!=((Label)o).label ) {
            return false;
        }
		if ( label==SET ) {
			return this.labelSet.equals(((Label)o).labelSet);
		}
		return true;  // label values are same, so true
    }

    public int compareTo(Object o) {
        return this.label-((Label)o).label;
    }

    /** Predicates are lists of AST nodes from the NFA created from the
     *  grammar, but the same predicate could be cut/paste into multiple
     *  places in the grammar.  I must compare the text of all the
     *  predicates to truly answer whether {p1,p2} .equals {p1,p2}.
     *  Unfortunately, I cannot rely on the AST.equals() to work properly
     *  so I must do a brute force O(n^2) nested traversal of the Set
     *  doing a String compare.
     *
     *  At this point, Labels are not compared for equals when they are
     *  predicates, but here's the code for future use.
     */
    /*
    protected boolean predicatesEquals(Set others) {
        Iterator iter = semanticContext.iterator();
        while (iter.hasNext()) {
            AST predAST = (AST) iter.next();
            Iterator inner = semanticContext.iterator();
            while (inner.hasNext()) {
                AST otherPredAST = (AST) inner.next();
                if ( !predAST.getText().equals(otherPredAST.getText()) ) {
                    return false;
                }
            }
        }
        return true;
    }
      */

    public String toString() {
        switch (label) {
            case SET :
                return labelSet.toString();
            default :
                return String.valueOf(label);
        }
    }

    public String toString(Grammar g) {
        switch (label) {
            case SET :
                return labelSet.toString(g);
            default :
                return g.getTokenDisplayName(label);
        }
    }

    /*
    public String predicatesToString() {
        if ( semanticContext==NFAConfiguration.DEFAULT_CLAUSE_SEMANTIC_CONTEXT ) {
            return "!other preds";
        }
        StringBuffer buf = new StringBuffer();
        Iterator iter = semanticContext.iterator();
        while (iter.hasNext()) {
            AST predAST = (AST) iter.next();
            buf.append(predAST.getText());
            if ( iter.hasNext() ) {
                buf.append("&");
            }
        }
        return buf.toString();
    }
    */

	public static boolean intersect(Label label, Label edgeLabel) {
		boolean hasIntersection = false;
		boolean labelIsSet = label.isSet();
		boolean edgeIsSet = edgeLabel.isSet();
		if ( !labelIsSet && !edgeIsSet && edgeLabel.label==label.label ) {
			hasIntersection = true;
		}
		else if ( labelIsSet && edgeIsSet &&
				  !edgeLabel.getSet().and(label.getSet()).isNil() ) {
			hasIntersection = true;
		}
		else if ( labelIsSet && !edgeIsSet &&
				  label.getSet().member(edgeLabel.label) ) {
			hasIntersection = true;
		}
		else if ( !labelIsSet && edgeIsSet &&
				  edgeLabel.getSet().member(label.label) ) {
			hasIntersection = true;
		}
		return hasIntersection;
	}
}
