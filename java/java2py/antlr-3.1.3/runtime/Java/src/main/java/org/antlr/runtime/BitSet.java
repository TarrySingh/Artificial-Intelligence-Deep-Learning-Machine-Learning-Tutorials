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
package org.antlr.runtime;

import java.util.List;

/**A stripped-down version of org.antlr.misc.BitSet that is just
 * good enough to handle runtime requirements such as FOLLOW sets
 * for automatic error recovery.
 */
public class BitSet implements Cloneable {
    protected final static int BITS = 64;    // number of bits / long
    protected final static int LOG_BITS = 6; // 2^6 == 64

    /* We will often need to do a mod operator (i mod nbits).  Its
     * turns out that, for powers of two, this mod operation is
     * same as (i & (nbits-1)).  Since mod is slow, we use a
     * precomputed mod mask to do the mod instead.
     */
    protected final static int MOD_MASK = BITS - 1;

    /** The actual data bits */
    protected long bits[];

    /** Construct a bitset of size one word (64 bits) */
    public BitSet() {
        this(BITS);
    }

    /** Construction from a static array of longs */
    public BitSet(long[] bits_) {
        bits = bits_;
    }

	/** Construction from a list of integers */
	public BitSet(List items) {
		this();
		for (int i = 0; i < items.size(); i++) {
			Integer v = (Integer) items.get(i);
			add(v.intValue());
		}
	}

    /** Construct a bitset given the size
     * @param nbits The size of the bitset in bits
     */
    public BitSet(int nbits) {
        bits = new long[((nbits - 1) >> LOG_BITS) + 1];
    }

	public static BitSet of(int el) {
		BitSet s = new BitSet(el + 1);
		s.add(el);
		return s;
	}

	public static BitSet of(int a, int b) {
		BitSet s = new BitSet(Math.max(a,b)+1);
		s.add(a);
		s.add(b);
		return s;
	}

	public static BitSet of(int a, int b, int c) {
		BitSet s = new BitSet();
		s.add(a);
		s.add(b);
		s.add(c);
		return s;
	}

	public static BitSet of(int a, int b, int c, int d) {
		BitSet s = new BitSet();
		s.add(a);
		s.add(b);
		s.add(c);
		s.add(d);
		return s;
	}

	/** return this | a in a new set */
	public BitSet or(BitSet a) {
		if ( a==null ) {
			return this;
		}
		BitSet s = (BitSet)this.clone();
		s.orInPlace(a);
		return s;
	}

	/** or this element into this set (grow as necessary to accommodate) */
	public void add(int el) {
		int n = wordNumber(el);
		if (n >= bits.length) {
			growToInclude(el);
		}
		bits[n] |= bitMask(el);
	}

	/**
	 * Grows the set to a larger number of bits.
	 * @param bit element that must fit in set
	 */
	public void growToInclude(int bit) {
		int newSize = Math.max(bits.length << 1, numWordsToHold(bit));
		long newbits[] = new long[newSize];
		System.arraycopy(bits, 0, newbits, 0, bits.length);
		bits = newbits;
	}

	public void orInPlace(BitSet a) {
		if ( a==null ) {
			return;
		}
		// If this is smaller than a, grow this first
		if (a.bits.length > bits.length) {
			setSize(a.bits.length);
		}
		int min = Math.min(bits.length, a.bits.length);
		for (int i = min - 1; i >= 0; i--) {
			bits[i] |= a.bits[i];
		}
	}

	/**
	 * Sets the size of a set.
	 * @param nwords how many words the new set should be
	 */
	private void setSize(int nwords) {
		long newbits[] = new long[nwords];
		int n = Math.min(nwords, bits.length);
		System.arraycopy(bits, 0, newbits, 0, n);
		bits = newbits;
	}

    private final static long bitMask(int bitNumber) {
        int bitPosition = bitNumber & MOD_MASK; // bitNumber mod BITS
        return 1L << bitPosition;
    }

    public Object clone() {
        BitSet s;
        try {
            s = (BitSet)super.clone();
            s.bits = new long[bits.length];
            System.arraycopy(bits, 0, s.bits, 0, bits.length);
        }
        catch (CloneNotSupportedException e) {
            throw new InternalError();
        }
        return s;
    }

    public int size() {
        int deg = 0;
        for (int i = bits.length - 1; i >= 0; i--) {
            long word = bits[i];
            if (word != 0L) {
                for (int bit = BITS - 1; bit >= 0; bit--) {
                    if ((word & (1L << bit)) != 0) {
                        deg++;
                    }
                }
            }
        }
        return deg;
    }

    public boolean equals(Object other) {
        if ( other == null || !(other instanceof BitSet) ) {
            return false;
        }

        BitSet otherSet = (BitSet)other;

        int n = Math.min(this.bits.length, otherSet.bits.length);

        // for any bits in common, compare
        for (int i=0; i<n; i++) {
            if (this.bits[i] != otherSet.bits[i]) {
                return false;
            }
        }

        // make sure any extra bits are off

        if (this.bits.length > n) {
            for (int i = n+1; i<this.bits.length; i++) {
                if (this.bits[i] != 0) {
                    return false;
                }
            }
        }
        else if (otherSet.bits.length > n) {
            for (int i = n+1; i<otherSet.bits.length; i++) {
                if (otherSet.bits[i] != 0) {
                    return false;
                }
            }
        }

        return true;
    }

    public boolean member(int el) {
		if ( el<0 ) {
			return false;
		}
        int n = wordNumber(el);
        if (n >= bits.length) return false;
        return (bits[n] & bitMask(el)) != 0;
    }

	// remove this element from this set
	public void remove(int el) {
		int n = wordNumber(el);
		if (n < bits.length) {
			bits[n] &= ~bitMask(el);
		}
	}

    public boolean isNil() {
        for (int i = bits.length - 1; i >= 0; i--) {
            if (bits[i] != 0) return false;
        }
        return true;
    }

    private final int numWordsToHold(int el) {
        return (el >> LOG_BITS) + 1;
    }

    public int numBits() {
        return bits.length << LOG_BITS; // num words * bits per word
    }

    /** return how much space is being used by the bits array not
     *  how many actually have member bits on.
     */
    public int lengthInLongWords() {
        return bits.length;
    }

    /**Is this contained within a? */
    /*
	public boolean subset(BitSet a) {
        if (a == null || !(a instanceof BitSet)) return false;
        return this.and(a).equals(this);
    }
	*/

    public int[] toArray() {
        int[] elems = new int[size()];
        int en = 0;
        for (int i = 0; i < (bits.length << LOG_BITS); i++) {
            if (member(i)) {
                elems[en++] = i;
            }
        }
        return elems;
    }

    public long[] toPackedArray() {
        return bits;
    }

	private final static int wordNumber(int bit) {
		return bit >> LOG_BITS; // bit / BITS
	}

	public String toString() {
		return toString(null);
	}

	public String toString(String[] tokenNames) {
		StringBuffer buf = new StringBuffer();
		String separator = ",";
		boolean havePrintedAnElement = false;
		buf.append('{');

		for (int i = 0; i < (bits.length << LOG_BITS); i++) {
			if (member(i)) {
				if (i > 0 && havePrintedAnElement ) {
					buf.append(separator);
				}
				if ( tokenNames!=null ) {
					buf.append(tokenNames[i]);
				}
				else {
					buf.append(i);
				}
				havePrintedAnElement = true;
			}
		}
		buf.append('}');
		return buf.toString();
	}


}
