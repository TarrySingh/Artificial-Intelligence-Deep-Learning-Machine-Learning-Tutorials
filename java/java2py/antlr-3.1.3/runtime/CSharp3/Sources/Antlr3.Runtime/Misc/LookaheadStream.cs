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

namespace Antlr.Runtime.Misc
{

    /** <summary>
     *  A lookahead queue that knows how to mark/release locations
     *  in the buffer for backtracking purposes. Any markers force the FastQueue
     *  superclass to keep all tokens until no more markers; then can reset
     *  to avoid growing a huge buffer.
     *  </summary>
     */
    public abstract class LookaheadStream<T>
        : FastQueue<T>
        where T : class
    {
        public const int UninitializedEofElementIndex = int.MaxValue;

        /** <summary>Set to buffer index of eof when nextElement returns eof</summary> */
        int _eofElementIndex = UninitializedEofElementIndex;

        /** <summary>Returned by nextElement upon end of stream; we add to buffer also</summary> */
        T _eof = null;

        /** <summary>Track the last mark() call result value for use in rewind().</summary> */
        int _lastMarker;

        /** <summary>tracks how deep mark() calls are nested</summary> */
        int _markDepth;

        public LookaheadStream( T eof )
        {
            this._eof = eof;
        }

        public T Eof
        {
            get
            {
                return _eof;
            }
            protected set
            {
                _eof = value;
            }
        }

        protected override void Clear()
        {
            _eofElementIndex = UninitializedEofElementIndex;
            base.Clear();
        }

        /** <summary>
         *  Implement nextElement to supply a stream of elements to this
         *  lookahead buffer.  Return eof upon end of the stream we're pulling from.
         *  </summary>
         */
        public abstract T NextElement();

        /** <summary>Get and remove first element in queue; override FastQueue.remove()</summary> */
        public override T Dequeue()
        {
            T o = this[0];
            _p++;
            // have we hit end of buffer and not backtracking?
            if ( _p == _data.Count && _markDepth == 0 )
            {
                // if so, it's an opportunity to start filling at index 0 again
                Clear(); // size goes to 0, but retains memory
            }
            return o;
        }

        /** <summary>Make sure we have at least one element to remove, even if EOF</summary> */
        public virtual void Consume()
        {
            Sync( 1 );
            Dequeue();
        }

        /** <summary>
         *  Make sure we have 'need' elements from current position p. Last valid
         *  p index is data.size()-1.  p+need-1 is the data index 'need' elements
         *  ahead.  If we need 1 element, (p+1-1)==p must be &lt; data.size().
         *  </summary>
         */
        public virtual void Sync( int need )
        {
            int n = ( _p + need - 1 ) - _data.Count + 1; // how many more elements we need?
            if ( n > 0 )
                Fill( n );                 // out of elements?
        }

        /** <summary>add n elements to buffer</summary> */
        public virtual void Fill( int n )
        {
            for ( int i = 1; i <= n; i++ )
            {
                T o = NextElement();
                if ( o == _eof )
                {
                    _data.Add( _eof );
                    _eofElementIndex = _data.Count - 1;
                }
                else
                    _data.Add( o );
            }
        }

        /** <summary>Size of entire stream is unknown; we only know buffer size from FastQueue</summary> */
        public override int Count
        {
            get
            {
                throw new System.NotSupportedException( "streams are of unknown size" );
            }
        }

        public virtual object LT( int k )
        {
            if ( k == 0 )
            {
                return null;
            }
            if ( k < 0 )
            {
                return LB( -k );
            }
            //System.out.print("LT(p="+p+","+k+")=");
            if ( ( _p + k - 1 ) >= _eofElementIndex )
            {
                // move to super.LT
                return _eof;
            }
            Sync( k );
            return this[k - 1];
        }

        /** <summary>Look backwards k nodes</summary> */
        protected virtual object LB( int k )
        {
            if ( k == 0 )
            {
                return null;
            }
            if ( ( _p - k ) < 0 )
            {
                return null;
            }
            return this[-k];
        }

        public virtual object GetCurrentSymbol()
        {
            return LT( 1 );
        }

        public virtual int Index
        {
            get
            {
                return _p;
            }
        }

        public virtual int Mark()
        {
            _markDepth++;
            _lastMarker = Index;
            return _lastMarker;
        }

        public virtual void Release( int marker )
        {
            _markDepth--;
        }

        public virtual void Rewind( int marker )
        {
            Seek( marker );
            Release( marker );
        }

        public virtual void Rewind()
        {
            Seek( _lastMarker );
        }

        /** <summary>
         *  Seek to a 0-indexed position within data buffer.  Can't handle
         *  case where you seek beyond end of existing buffer.  Normally used
         *  to seek backwards in the buffer. Does not force loading of nodes.
         *  </summary>
         */
        public virtual void Seek( int index )
        {
            _p = index;
        }
    }
}
