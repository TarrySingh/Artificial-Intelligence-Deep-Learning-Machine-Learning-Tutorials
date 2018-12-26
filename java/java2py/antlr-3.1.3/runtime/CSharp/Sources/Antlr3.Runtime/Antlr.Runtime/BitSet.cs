/*
[The "BSD licence"]
Copyright (c) 2005-2007 Kunle Odutola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

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


namespace Antlr.Runtime
{
    using System;
    using IList         = System.Collections.IList;
    using StringBuilder = System.Text.StringBuilder;
	
	/// <summary>
    /// A stripped-down version of org.antlr.misc.BitSet that is just
	/// good enough to handle runtime requirements such as FOLLOW sets
	/// for automatic error recovery.
	/// </summary>
	public class BitSet : ICloneable
    {
        #region Constructors

        /// <summary>Construct a bitset of size one word (64 bits) </summary>
        public BitSet()
            : this(BITS)
        {
        }

        /// <summary>Construction from a static array of ulongs </summary>
        public BitSet(ulong[] bits_)
        {
            bits = bits_;
        }

        /// <summary>Construction from a list of integers </summary>
        public BitSet(IList items)
        	: this(BITS)
        {
            for (int i = 0; i < items.Count; i++)
            {
                int v = (int)items[i];
                Add(v);
            }
        }

        /// <summary>Construct a bitset given the size</summary>
        /// <param name="nbits">The size of the bitset in bits</param>
        public BitSet(int nbits)
        {
            bits = new ulong[((nbits - 1) >> LOG_BITS) + 1];
        }
		
        #endregion

        #region Public API

        public static BitSet Of(int el)
        {
            BitSet s = new BitSet(el + 1);
            s.Add(el);
            return s;
        }

        public static BitSet Of(int a, int b)
        {
            BitSet s = new BitSet(Math.Max(a, b) + 1);
            s.Add(a);
            s.Add(b);
            return s;
        }

        public static BitSet Of(int a, int b, int c)
        {
            BitSet s = new BitSet();
            s.Add(a);
            s.Add(b);
            s.Add(c);
            return s;
        }

        public static BitSet Of(int a, int b, int c, int d)
        {
            BitSet s = new BitSet();
            s.Add(a);
            s.Add(b);
            s.Add(c);
            s.Add(d);
            return s;
        }

        /// <summary>return "this | a" in a new set </summary>
        public virtual BitSet Or(BitSet a)
        {
            if (a == null)
            {
                return this;
            }
            BitSet s = (BitSet)this.Clone();
            s.OrInPlace(a);
            return s;
        }

        /// <summary>Or this element into this set (grow as necessary to accommodate)</summary>
        public virtual void Add(int el)
        {
            int n = WordNumber(el);
            if (n >= bits.Length)
            {
                GrowToInclude(el);
            }
            bits[n] |= BitMask(el);
        }

        /// <summary> Grows the set to a larger number of bits.</summary>
        /// <param name="bit">element that must fit in set
        /// </param>
        public virtual void GrowToInclude(int bit)
        {
            int newSize = Math.Max(bits.Length << 1, NumWordsToHold(bit));
            ulong[] newbits = new ulong[newSize];
            Array.Copy(bits, 0, newbits, 0, bits.Length);
            bits = newbits;
        }

        public virtual void OrInPlace(BitSet a)
        {
            if (a == null)
            {
                return;
            }
            // If this is smaller than a, grow this first
            if (a.bits.Length > bits.Length)
            {
                SetSize(a.bits.Length);
            }
            int min = Math.Min(bits.Length, a.bits.Length);
            for (int i = min - 1; i >= 0; i--)
            {
                bits[i] |= a.bits[i];
            }
        }

        virtual public bool Nil
        {
            get
            {
                for (int i = bits.Length - 1; i >= 0; i--)
                {
                    if (bits[i] != 0)
                        return false;
                }
                return true;
            }

        }

        public virtual object Clone()
        {
            BitSet s;
            try
            {
                s = (BitSet)MemberwiseClone();
                s.bits = new ulong[bits.Length];
                Array.Copy(bits, 0, s.bits, 0, bits.Length);
            }
            catch (System.Exception e)
            {
                throw new InvalidOperationException("Unable to clone BitSet", e);
            }
            return s;
        }

        public virtual int Count
        {
	    get
	    {
		int deg = 0;
		for (int i = bits.Length - 1; i >= 0; i--)
		{
		    ulong word = bits[i];
		    if (word != 0L)
		    {
			for (int bit = BITS - 1; bit >= 0; bit--)
			{
			    if ((word & (1UL << bit)) != 0)
			    {
				deg++;
			    }
			}
		    }
		}
		return deg;
		}
        }

        public virtual bool Member(int el)
        {
            if (el < 0)
            {
                return false;
            }
            int n = WordNumber(el);
            if (n >= bits.Length)
                return false;
            return (bits[n] & BitMask(el)) != 0;
        }

        // remove this element from this set
        public virtual void Remove(int el)
        {
            int n = WordNumber(el);
            if (n < bits.Length)
            {
                bits[n] &= ~BitMask(el);
            }
        }

        public virtual int NumBits()
        {
            return bits.Length << LOG_BITS; // num words * bits per word
        }

        /// <summary>return how much space is being used by the bits array not
        /// how many actually have member bits on.
        /// </summary>
        public virtual int LengthInLongWords()
        {
            return bits.Length;
        }

        public virtual int[] ToArray()
        {
            int[] elems = new int[Count];
            int en = 0;
            for (int i = 0; i < (bits.Length << LOG_BITS); i++)
            {
                if (Member(i))
                {
                    elems[en++] = i;
                }
            }
            return elems;
        }

        public virtual ulong[] ToPackedArray()
        {
            return bits;
        }

        private static int WordNumber(int bit)
        {
            return bit >> LOG_BITS; // bit / BITS
        }

        public override string ToString()
        {
            return ToString(null);
        }

        public virtual string ToString(string[] tokenNames)
        {
            StringBuilder buf = new StringBuilder();
            string separator = ",";
            bool havePrintedAnElement = false;

            buf.Append('{');

            for (int i = 0; i < (bits.Length << LOG_BITS); i++)
            {
                if (Member(i))
                {
                    if (i > 0 && havePrintedAnElement)
                    {
                        buf.Append(separator);
                    }
                    if (tokenNames != null)
                    {
                        buf.Append(tokenNames[i]);
                    }
                    else
                    {
                        buf.Append(i);
                    }
                    havePrintedAnElement = true;
                }
            }
            buf.Append('}');
            return buf.ToString();
        }

        public override bool Equals(object other)
        {
            if (other == null || !(other is BitSet))
            {
                return false;
            }

            BitSet otherSet = (BitSet)other;

            int n = Math.Min(this.bits.Length, otherSet.bits.Length);

            // for any bits in common, compare
            for (int i = 0; i < n; i++)
            {
                if (this.bits[i] != otherSet.bits[i])
                {
                    return false;
                }
            }

            // make sure any extra bits are off

            if (this.bits.Length > n)
            {
                for (int i = n + 1; i < this.bits.Length; i++)
                {
                    if (this.bits[i] != 0)
                    {
                        return false;
                    }
                }
            }
            else if (otherSet.bits.Length > n)
            {
                for (int i = n + 1; i < otherSet.bits.Length; i++)
                {
                    if (otherSet.bits[i] != 0)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        #endregion

        #region Data Members

        protected internal const int BITS = 64; // number of bits / ulong
        protected internal const int LOG_BITS = 6; // 2^6 == 64

        ///<summary> We will often need to do a mod operator (i mod nbits).
        /// Its turns out that, for powers of two, this mod operation is
		///  same as <![CDATA[(i & (nbits-1))]]>.  Since mod is slow, we use a precomputed 
        /// mod mask to do the mod instead.
        /// </summary>
        protected internal static readonly int MOD_MASK = BITS - 1;

        /// <summary>The actual data bits </summary>
        protected internal ulong[] bits;

        #endregion

        #region Private API

        private static ulong BitMask(int bitNumber)
        {
            int bitPosition = bitNumber & MOD_MASK; // bitNumber mod BITS
            return 1UL << bitPosition;
        }

        /// <summary> Sets the size of a set.</summary>
		/// <param name="nwords">how many words the new set should be
		/// </param>
        private void SetSize(int nwords)
        {
            ulong[] newbits = new ulong[nwords];
            int n = Math.Min(nwords, bits.Length);
            Array.Copy(bits, 0, newbits, 0, n);
            bits = newbits;
        }

        private int NumWordsToHold(int el)
		{
			return (el >> LOG_BITS) + 1;
        }

        #endregion
    }
}