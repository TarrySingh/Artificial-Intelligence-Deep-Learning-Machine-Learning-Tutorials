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
package org.antlr.runtime {

	/**A stripped-down version of org.antlr.misc.BitSet that is just
	 * good enough to handle runtime requirements such as FOLLOW sets
	 * for automatic error recovery.
	 */
	public class BitSet {
	    protected static const BITS:uint = 32;    // number of bits / int
	    protected static const LOG_BITS:uint = 5; // 2^5 == 32
	
	    /* We will often need to do a mod operator (i mod nbits).  Its
	     * turns out that, for powers of two, this mod operation is
	     * same as (i & (nbits-1)).  Since mod is slow, we use a
	     * precomputed mod mask to do the mod instead.
	     */
	    protected static const MOD_MASK:uint = BITS - 1;

	    /** The actual data bits */
	    protected var bits:Array;
	
	    /** Construction from a static array of longs */
	    public function BitSet(bits:Array = null) {
	        if (bits == null) {
	            this.bits = new Array();
	        }
	        else {
    	        this.bits = new Array(bits.length);
    			for (var i:int = 0; i < bits.length; i++) {
    				this.bits[i] = bits[i];
    	    	}
	        }
	    }
	
		public static function of(... args):BitSet {
			var s:BitSet = new BitSet();
			for (var i:int = 0; i < args.length; i++) {
				s.add(args[i]);
			}
			return s;
		}
	
		/** return this | a in a new set */
		public function or(a:BitSet):BitSet {
			if ( a==null ) {
				return this;
			}
			var s:BitSet = this.clone();
			s.orInPlace(a);
			return s;
		}
	
		/** or this element into this set (grow as necessary to accommodate) */
		public function add(el:int):void {
			var n:int = wordNumber(el);
			if (n >= bits.length) {
				growToInclude(el);
			}
			bits[n] |= bitMask(el);
		}
	
		/**
		 * Grows the set to a larger number of bits.
		 * @param bit element that must fit in set
		 */
		public function growToInclude(bit:int):void {
			var newSize:int = Math.max(bits.length << 1, numWordsToHold(bit));
			bits.length = newSize;
		}
	
		public function orInPlace(a:BitSet):void {
			if ( a==null ) {
				return;
			}
			// If this is smaller than a, grow this first
			if (a.bits.length > bits.length) {
				this.bits.length = a.bits.length;
			}
			var min:int = Math.min(bits.length, a.bits.length);
			for (var i:int = min - 1; i >= 0; i--) {
				bits[i] |= a.bits[i];
			}
		}
	
		/**
		 * Sets the size of a set.
		 * @param nwords how many words the new set should be
		 */
		private function set size(nwords:int):void {
			bits.length = nwords;
		}
	
	    private static function bitMask(bitNumber:int):int {
	        var bitPosition:int = bitNumber & MOD_MASK; // bitNumber mod BITS
	        return 1 << bitPosition;
	    }
	
	    public function clone():BitSet {
	        var s:BitSet = new BitSet(bits);
			return s;
	    }
	
	    public function get size():int {
	        var deg:uint = 0;
	        for (var i:int = bits.length - 1; i >= 0; i--) {
	            var word:uint = bits[i];
	            if (word != 0) {
	                for (var bit:int = BITS - 1; bit >= 0; bit--) {
	                    if ((word & (1 << bit)) != 0) {
	                        deg++;
	                    }
	                }
	            }
	        }
	        return deg;
	    }
	
	    public function equals(other:Object):Boolean {
	        if ( other == null || !(other is BitSet) ) {
	            return false;
	        }
	
	        var otherSet:BitSet = BitSet(other);
	
	        var n:int = Math.min(this.bits.length, otherSet.bits.length);
	
	        // for any bits in common, compare
	        for (var i:int=0; i<n; i++) {
	            if (this.bits[i] != otherSet.bits[i]) {
	                return false;
	            }
	        }
	
	        // make sure any extra bits are off
	
	        if (this.bits.length > n) {
	            for (i = n+1; i<this.bits.length; i++) {
	                if (this.bits[i] != 0) {
	                    return false;
	                }
	            }
	        }
	        else if (otherSet.bits.length > n) {
	            for (i = n+1; i<otherSet.bits.length; i++) {
	                if (otherSet.bits[i] != 0) {
	                    return false;
	                }
	            }
	        }
	
	        return true;
	    }
	
	    public function member(el:int):Boolean {
			if ( el<0 ) {
				return false;
			}
	        var n:int = wordNumber(el);
	        if (n >= bits.length) return false;
	        return (bits[n] & bitMask(el)) != 0;
	    }
	
		// remove this element from this set
		public function remove(el:int):void {
			var n:int = wordNumber(el);
			if (n < bits.length) {
				bits[n] &= ~bitMask(el);
			}
		}
	
	    public function get isNil():Boolean {
	        for (var i:int = bits.length - 1; i >= 0; i--) {
	            if (bits[i] != 0) return false;
	        }
	        return true;
	    }
	
	    private final function numWordsToHold(el:int):int {
	        return (el >> LOG_BITS) + 1;
	    }
	
	    public function get numBits():int {
	        return bits.length << LOG_BITS; // num words * bits per word
	    }
	
	    /** return how much space is being used by the bits array not
	     *  how many actually have member bits on.
	     */
	    public function get lengthInLongWords():int {
	        return bits.length;
	    }
	
	    public function toArray():Array {
	        var elems:Array = new Array[this.bits.length];
	        var en:int = 0;
	        for (var i:int = 0; i < (bits.length << LOG_BITS); i++) {
	            if (member(i)) {
	                elems[en++] = i;
	            }
	        }
	        return elems;
	    }
	
	    public function toPackedArray():Array {
	        return bits;
	    }
	
		private static function wordNumber(bit:uint):uint {
			return bit >> LOG_BITS; // bit / BITS
		}
	
		public function toString():String {
			return toStringFromTokens(null);
		}
	
		public function toStringFromTokens(tokenNames:Array):String {
			var buf:String = "";
			const separator:String = ",";
			var havePrintedAnElement:Boolean = false;
			buf = buf + '{';
	
			for (var i:int = 0; i < (bits.length << LOG_BITS); i++) {
				if (member(i)) {
					if (i > 0 && havePrintedAnElement ) {
						buf += separator;
					}
					if ( tokenNames!=null ) {
						buf += tokenNames[i];
					}
					else {
						buf += i;
					}
					havePrintedAnElement = true;
				}
			}
			buf += '}';
			return buf;
		}
	
	}
}