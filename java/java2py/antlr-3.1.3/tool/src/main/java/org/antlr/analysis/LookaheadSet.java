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

import org.antlr.misc.IntervalSet;
import org.antlr.misc.IntSet;
import org.antlr.tool.Grammar;

/** An LL(1) lookahead set; contains a set of token types and a "hasEOF"
 *  condition when the set contains EOF.  Since EOF is -1 everywhere and -1
 *  cannot be stored in my BitSet, I set a condition here.  There may be other
 *  reasons in the future to abstract a LookaheadSet over a raw BitSet.
 */
public class LookaheadSet {
	public IntervalSet tokenTypeSet;

	public LookaheadSet() {
		tokenTypeSet = new IntervalSet();
	}

	public LookaheadSet(IntSet s) {
		this();
		tokenTypeSet.addAll(s);
	}

	public LookaheadSet(int atom) {
		tokenTypeSet = IntervalSet.of(atom);
	}

    public LookaheadSet(LookaheadSet other) {
        this();
        this.tokenTypeSet.addAll(other.tokenTypeSet);
    }

    public void orInPlace(LookaheadSet other) {
		this.tokenTypeSet.addAll(other.tokenTypeSet);
	}

	public LookaheadSet or(LookaheadSet other) {
		return new LookaheadSet(tokenTypeSet.or(other.tokenTypeSet));
	}

	public LookaheadSet subtract(LookaheadSet other) {
		return new LookaheadSet(this.tokenTypeSet.subtract(other.tokenTypeSet));
	}

	public boolean member(int a) {
		return tokenTypeSet.member(a);
	}

	public LookaheadSet intersection(LookaheadSet s) {
		IntSet i = this.tokenTypeSet.and(s.tokenTypeSet);
		LookaheadSet intersection = new LookaheadSet(i);
		return intersection;
	}

	public boolean isNil() {
		return tokenTypeSet.isNil();
	}

	public void remove(int a) {
		tokenTypeSet = (IntervalSet)tokenTypeSet.subtract(IntervalSet.of(a));
	}

	public int hashCode() {
		return tokenTypeSet.hashCode();
	}

	public boolean equals(Object other) {
		return tokenTypeSet.equals(((LookaheadSet)other).tokenTypeSet);
	}

	public String toString(Grammar g) {
		if ( tokenTypeSet==null ) {
			return "";
		}
		String r = tokenTypeSet.toString(g);
		return r;
	}

	public String toString() {
		return toString(null);
	}
}
