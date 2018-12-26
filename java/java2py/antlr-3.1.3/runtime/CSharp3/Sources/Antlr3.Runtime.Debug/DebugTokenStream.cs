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

namespace Antlr.Runtime.Debug
{

    public class DebugTokenStream : ITokenStream
    {
        protected IDebugEventListener dbg;
        public ITokenStream input;
        protected bool initialStreamState = true;

        /** <summary>Track the last mark() call result value for use in rewind().</summary> */
        protected int lastMarker;

        public DebugTokenStream( ITokenStream input, IDebugEventListener dbg )
        {
            this.input = input;
            DebugListener = dbg;
            // force TokenStream to get at least first valid token
            // so we know if there are any hidden tokens first in the stream
            input.LT( 1 );
        }

        #region Properties
        public virtual int Index
        {
            get
            {
                return input.Index;
            }
        }
        public virtual IDebugEventListener DebugListener
        {
            get
            {
                return dbg;
            }
            set
            {
                dbg = value;
            }
        }
        #endregion

        [System.Obsolete]
        public void SetDebugListener( IDebugEventListener dbg )
        {
            DebugListener = dbg;
        }

        public virtual void Consume()
        {
            if ( initialStreamState )
            {
                ConsumeInitialHiddenTokens();
            }
            int a = input.Index;
            IToken t = input.LT( 1 );
            input.Consume();
            int b = input.Index;
            dbg.ConsumeToken( t );
            if ( b > a + 1 )
            {
                // then we consumed more than one token; must be off channel tokens
                for ( int i = a + 1; i < b; i++ )
                {
                    dbg.ConsumeHiddenToken( input.Get( i ) );
                }
            }
        }

        /** <summary>Consume all initial off-channel tokens</summary> */
        protected virtual void ConsumeInitialHiddenTokens()
        {
            int firstOnChannelTokenIndex = input.Index;
            for ( int i = 0; i < firstOnChannelTokenIndex; i++ )
            {
                dbg.ConsumeHiddenToken( input.Get( i ) );
            }
            initialStreamState = false;
        }

        public virtual IToken LT( int i )
        {
            if ( initialStreamState )
            {
                ConsumeInitialHiddenTokens();
            }
            dbg.LT( i, input.LT( i ) );
            return input.LT( i );
        }

        public virtual int LA( int i )
        {
            if ( initialStreamState )
            {
                ConsumeInitialHiddenTokens();
            }
            dbg.LT( i, input.LT( i ) );
            return input.LA( i );
        }

        public virtual IToken Get( int i )
        {
            return input.Get( i );
        }

        public virtual int Mark()
        {
            lastMarker = input.Mark();
            dbg.Mark( lastMarker );
            return lastMarker;
        }

        public virtual void Rewind( int marker )
        {
            dbg.Rewind( marker );
            input.Rewind( marker );
        }

        public virtual void Rewind()
        {
            dbg.Rewind();
            input.Rewind( lastMarker );
        }

        public virtual void Release( int marker )
        {
        }

        public virtual void Seek( int index )
        {
            // TODO: implement seek in dbg interface
            // db.seek(index);
            input.Seek( index );
        }

        public virtual int Size()
        {
            return input.Size();
        }

        public virtual ITokenSource TokenSource
        {
            get
            {
                return input.TokenSource;
            }
        }

        public virtual string SourceName
        {
            get
            {
                return TokenSource.SourceName;
            }
        }

        public override string ToString()
        {
            return input.ToString();
        }

        public virtual string ToString( int start, int stop )
        {
            return input.ToString( start, stop );
        }

        public virtual string ToString( IToken start, IToken stop )
        {
            return input.ToString( start, stop );
        }
    }
}
