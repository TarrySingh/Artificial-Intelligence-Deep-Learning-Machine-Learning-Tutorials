/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008-2009 Sam Harwell, Pixel Mine, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace Antlr.Runtime
{
    using System.Collections.Generic;

    using Array = System.Array;
    using ICloneable = System.ICloneable;
    using Math = System.Math;
    using StringBuilder = System.Text.StringBuilder;

    /** <summary>
     *  A stripped-down version of org.antlr.misc.BitSet that is just
     *  good enough to handle runtime requirements such as FOLLOW sets
     *  for automatic error recovery.
     *  </summary>
     */
    [System.Serializable]
    public class BitSet : ICloneable
    {
        protected const int BITS = 64;    // number of bits / long
        protected const int LOG_BITS = 6; // 2^6 == 64

        /** <summary>
         *  We will often need to do a mod operator (i mod nbits).  Its
         *  turns out that, for powers of two, this mod operation is
         *  same as (i & (nbits-1)).  Since mod is slow, we use a
         *  precomputed mod mask to do the mod instead.
         *  </summary>
         */
        protected const int MOD_MASK = BITS - 1;

        /** <summary>The actual data bits</summary> */
        ulong[] _bits;

        /** <summary>Construct a bitset of size one word (64 bits)</summary> */
        public BitSet()
            : this( BITS )
        {
        }

        /** <summary>Construction from a static array of longs</summary> */
        public BitSet( ulong[] bits )
        {
            _bits = bits;
        }

        /** <summary>Construction from a list of integers</summary> */
        public BitSet( IEnumerable<int> items )
            : this()
        {
            foreach ( int i in items )
                Add( i );
        }

        /** <summary>Construct a bitset given the size</summary>
         *  <param name="nbits">The size of the bitset in bits</param>
         */
        public BitSet( int nbits )
        {
            _bits = new ulong[( ( nbits - 1 ) >> LOG_BITS ) + 1];
        }

        public static BitSet Of( int el )
        {
            BitSet s = new BitSet( el + 1 );
            s.Add( el );
            return s;
        }

        public static BitSet Of( int a, int b )
        {
            BitSet s = new BitSet( Math.Max( a, b ) + 1 );
            s.Add( a );
            s.Add( b );
            return s;
        }

        public static BitSet Of( int a, int b, int c )
        {
            BitSet s = new BitSet();
            s.Add( a );
            s.Add( b );
            s.Add( c );
            return s;
        }

        public static BitSet Of( int a, int b, int c, int d )
        {
            BitSet s = new BitSet();
            s.Add( a );
            s.Add( b );
            s.Add( c );
            s.Add( d );
            return s;
        }

        /** <summary>return this | a in a new set</summary> */
        public virtual BitSet Or( BitSet a )
        {
            if ( a == null )
            {
                return this;
            }
            BitSet s = (BitSet)this.Clone();
            s.OrInPlace( a );
            return s;
        }

        /** <summary>or this element into this set (grow as necessary to accommodate)</summary> */
        public virtual void Add( int el )
        {
            int n = WordNumber( el );
            if ( n >= _bits.Length )
            {
                GrowToInclude( el );
            }
            _bits[n] |= BitMask( el );
        }

        /** <summary>Grows the set to a larger number of bits.</summary>
         *  <param name="bit">element that must fit in set</param>
         */
        public virtual void GrowToInclude( int bit )
        {
            int newSize = Math.Max( _bits.Length << 1, NumWordsToHold( bit ) );
            ulong[] newbits = new ulong[newSize];
            Array.Copy( _bits, newbits, _bits.Length );
            _bits = newbits;
        }

        public virtual void OrInPlace( BitSet a )
        {
            if ( a == null )
            {
                return;
            }
            // If this is smaller than a, grow this first
            if ( a._bits.Length > _bits.Length )
            {
                SetSize( a._bits.Length );
            }
            int min = Math.Min( _bits.Length, a._bits.Length );
            for ( int i = min - 1; i >= 0; i-- )
            {
                _bits[i] |= a._bits[i];
            }
        }

        /** <summary>Sets the size of a set.</summary>
         *  <param name="nwords">how many words the new set should be</param>
         */
        private void SetSize( int nwords )
        {
            ulong[] newbits = new ulong[nwords];
            int n = Math.Min( nwords, _bits.Length );
            Array.Copy( _bits, newbits, n );
            _bits = newbits;
        }

        private static ulong BitMask( int bitNumber )
        {
            int bitPosition = bitNumber & MOD_MASK; // bitNumber mod BITS
            return 1UL << bitPosition;
        }

        public virtual object Clone()
        {
            return new BitSet( (ulong[])_bits.Clone() );
        }

        public virtual int Size()
        {
            int deg = 0;
            for ( int i = _bits.Length - 1; i >= 0; i-- )
            {
                ulong word = _bits[i];
                if ( word != 0L )
                {
                    for ( int bit = BITS - 1; bit >= 0; bit-- )
                    {
                        if ( ( word & ( 1UL << bit ) ) != 0 )
                        {
                            deg++;
                        }
                    }
                }
            }
            return deg;
        }

        public override int GetHashCode()
        {
            throw new System.NotImplementedException();
        }

        public override bool Equals( object other )
        {
            if ( other == null || !( other is BitSet ) )
            {
                return false;
            }

            BitSet otherSet = (BitSet)other;

            int n = Math.Min( this._bits.Length, otherSet._bits.Length );

            // for any bits in common, compare
            for ( int i = 0; i < n; i++ )
            {
                if ( this._bits[i] != otherSet._bits[i] )
                {
                    return false;
                }
            }

            // make sure any extra bits are off

            if ( this._bits.Length > n )
            {
                for ( int i = n + 1; i < this._bits.Length; i++ )
                {
                    if ( this._bits[i] != 0 )
                    {
                        return false;
                    }
                }
            }
            else if ( otherSet._bits.Length > n )
            {
                for ( int i = n + 1; i < otherSet._bits.Length; i++ )
                {
                    if ( otherSet._bits[i] != 0 )
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        public virtual bool Member( int el )
        {
            if ( el < 0 )
            {
                return false;
            }
            int n = WordNumber( el );
            if ( n >= _bits.Length )
                return false;
            return ( _bits[n] & BitMask( el ) ) != 0;
        }

        // remove this element from this set
        public virtual void Remove( int el )
        {
            int n = WordNumber( el );
            if ( n < _bits.Length )
            {
                _bits[n] &= ~BitMask( el );
            }
        }

        public virtual bool IsNil()
        {
            for ( int i = _bits.Length - 1; i >= 0; i-- )
            {
                if ( _bits[i] != 0 )
                    return false;
            }
            return true;
        }

        private int NumWordsToHold( int el )
        {
            return ( el >> LOG_BITS ) + 1;
        }

        public virtual int NumBits()
        {
            return _bits.Length << LOG_BITS; // num words * bits per word
        }

        /** <summary>return how much space is being used by the bits array not how many actually have member bits on.</summary> */
        public virtual int LengthInLongWords()
        {
            return _bits.Length;
        }

        /**Is this contained within a? */
        /*
        public boolean subset(BitSet a) {
            if (a == null || !(a instanceof BitSet)) return false;
            return this.and(a).equals(this);
        }
        */

        public virtual int[] ToArray()
        {
            int[] elems = new int[Size()];
            int en = 0;
            for ( int i = 0; i < ( _bits.Length << LOG_BITS ); i++ )
            {
                if ( Member( i ) )
                {
                    elems[en++] = i;
                }
            }
            return elems;
        }

        public virtual ulong[] ToPackedArray()
        {
            return _bits;
        }

        private static int WordNumber( int bit )
        {
            return bit >> LOG_BITS; // bit / BITS
        }

        public override string ToString()
        {
            return ToString( null );
        }

        public virtual string ToString( string[] tokenNames )
        {
            StringBuilder buf = new StringBuilder();
            string separator = ",";
            bool havePrintedAnElement = false;
            buf.Append( '{' );

            for ( int i = 0; i < ( _bits.Length << LOG_BITS ); i++ )
            {
                if ( Member( i ) )
                {
                    if ( i > 0 && havePrintedAnElement )
                    {
                        buf.Append( separator );
                    }
                    if ( tokenNames != null )
                    {
                        buf.Append( tokenNames[i] );
                    }
                    else
                    {
                        buf.Append( i );
                    }
                    havePrintedAnElement = true;
                }
            }
            buf.Append( '}' );
            return buf.ToString();
        }
    }
}
