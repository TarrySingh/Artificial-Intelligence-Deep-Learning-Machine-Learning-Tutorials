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

import java.util.AbstractList;

/** An ArrayList based upon int members.  Not quite a real implementation of a
 *  modifiable list as I don't do, for example, add(index,element).
 *  TODO: unused?
 */
public class IntArrayList extends AbstractList implements Cloneable {
	private static final int DEFAULT_CAPACITY = 10;
	protected int n = 0;
	protected int[] elements = null;

	public IntArrayList() {
		this(DEFAULT_CAPACITY);
	}

	public IntArrayList(int initialCapacity) {
		elements = new int[initialCapacity];
	}

	/** Set the ith element.  Like ArrayList, this does NOT affect size. */
	public int set(int i, int newValue) {
		if ( i>=n ) {
			setSize(i); // unlike definition of set in ArrayList, set size
		}
		int v = elements[i];
		elements[i] = newValue;
		return v;
	}

	public boolean add(int o) {
		if ( n>=elements.length ) {
			grow();
		}
		elements[n] = o;
		n++;
		return true;
	}

	public void setSize(int newSize) {
		if ( newSize>=elements.length ) {
            ensureCapacity(newSize);
		}
		n = newSize;
	}

	protected void grow() {
		ensureCapacity((elements.length * 3)/2 + 1);
	}

	public boolean contains(int v) {
		for (int i = 0; i < n; i++) {
			int element = elements[i];
			if ( element == v ) {
				return true;
			}
		}
		return false;
	}

	public void ensureCapacity(int newCapacity) {
		int oldCapacity = elements.length;
		if (n>=oldCapacity) {
			int oldData[] = elements;
			elements = new int[newCapacity];
			System.arraycopy(oldData, 0, elements, 0, n);
		}
	}

	public Object get(int i) {
		return Utils.integer(element(i));
	}

	public int element(int i) {
		return elements[i];
	}

	public int[] elements() {
		int[] a = new int[n];
		System.arraycopy(elements, 0, a, 0, n);
		return a;
	}

	public int size() {
		return n;
	}

    public int capacity() {
        return elements.length;
    }

	public boolean equals(Object o) {
        if ( o==null ) {
            return false;
        }
        IntArrayList other = (IntArrayList)o;
        if ( this.size()!=other.size() ) {
            return false;
        }
		for (int i = 0; i < n; i++) {
			if ( elements[i] != other.elements[i] ) {
				return false;
			}
		}
		return true;
	}

    public Object clone() throws CloneNotSupportedException {
		IntArrayList a = (IntArrayList)super.clone();
        a.n = this.n;
        System.arraycopy(this.elements, 0, a.elements, 0, this.elements.length);
        return a;
    }

	public String toString() {
		StringBuffer buf = new StringBuffer();
		for (int i = 0; i < n; i++) {
			if ( i>0 ) {
				buf.append(", ");
			}
			buf.append(elements[i]);
		}
		return buf.toString();
	}
}
