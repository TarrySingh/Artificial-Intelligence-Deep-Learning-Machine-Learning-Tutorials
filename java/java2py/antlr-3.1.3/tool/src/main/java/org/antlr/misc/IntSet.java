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
package org.antlr.misc;

import org.antlr.tool.Grammar;

import java.util.List;

/** A generic set of ints that has an efficient implementation, BitSet,
 *  which is a compressed bitset and is useful for ints that
 *  are small, for example less than 500 or so, and w/o many ranges.  For
 *  ranges with large values like unicode char sets, this is not very efficient.
 *  Consider using IntervalSet.  Not all methods in IntervalSet are implemented.
 *
 *  @see org.antlr.misc.BitSet
 *  @see org.antlr.misc.IntervalSet
 */
public interface IntSet {
    /** Add an element to the set */
    void add(int el);

    /** Add all elements from incoming set to this set.  Can limit
     *  to set of its own type.
     */
    void addAll(IntSet set);

    /** Return the intersection of this set with the argument, creating
     *  a new set.
     */
    IntSet and(IntSet a);

    IntSet complement(IntSet elements);

    IntSet or(IntSet a);

    IntSet subtract(IntSet a);

    /** Return the size of this set (not the underlying implementation's
     *  allocated memory size, for example).
     */
    int size();

    boolean isNil();

    boolean equals(Object obj);

    int getSingleElement();

    boolean member(int el);

    /** remove this element from this set */
    void remove(int el);

    List toList();

    String toString();

    String toString(Grammar g);
}
