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

    using StringBuilder = System.Text.StringBuilder;

    /** <summary>
     *  The most common stream of tokens is one where every token is buffered up
     *  and tokens are prefiltered for a certain channel (the parser will only
     *  see these tokens and cannot change the filter channel number during the
     *  parse).
     *  </summary>
     *
     *  <remarks>TODO: how to access the full token stream?  How to track all tokens matched per rule?</remarks>
     */
    [System.Serializable]
    public class CommonTokenStream : ITokenStream
    {
        [System.NonSerialized]
        protected ITokenSource tokenSource;

        /** <summary>
         *  Record every single token pulled from the source so we can reproduce
         *  chunks of it later.
         *  </summary>
         */
        protected List<IToken> tokens;

        /** <summary>Map<tokentype, channel> to override some Tokens' channel numbers</summary> */
        protected IDictionary<int, int> channelOverrideMap;

        /** <summary>Set<tokentype>; discard any tokens with this type</summary> */
        protected HashSet<int> discardSet;

        /** <summary>Skip tokens on any channel but this one; this is how we skip whitespace...</summary> */
        protected int channel = TokenConstants.DEFAULT_CHANNEL;

        /** <summary>By default, track all incoming tokens</summary> */
        protected bool discardOffChannelTokens = false;

        /** <summary>Track the last mark() call result value for use in rewind().</summary> */
        protected int lastMarker;

        /** <summary>
         *  The index into the tokens list of the current token (next token
         *  to consume).  p==-1 indicates that the tokens list is empty
         *  </summary>
         */
        protected int p = -1;

        public CommonTokenStream()
        {
            tokens = new List<IToken>( 500 );
        }

        public CommonTokenStream( ITokenSource tokenSource )
            : this()
        {
            this.tokenSource = tokenSource;
        }

        public CommonTokenStream( ITokenSource tokenSource, int channel )
            : this( tokenSource )
        {
            this.channel = channel;
        }

        public virtual int Index
        {
            get
            {
                return p;
            }
        }

        /** <summary>Reset this token stream by setting its token source.</summary> */
        public virtual void SetTokenSource( ITokenSource tokenSource )
        {
            this.tokenSource = tokenSource;
            tokens.Clear();
            p = -1;
            channel = TokenConstants.DEFAULT_CHANNEL;
        }

        /** <summary>
         *  Load all tokens from the token source and put in tokens.
         *  This is done upon first LT request because you might want to
         *  set some token type / channel overrides before filling buffer.
         *  </summary>
         */
        protected virtual void FillBuffer()
        {
            int index = 0;
            IToken t = tokenSource.NextToken();
            while ( t != null && t.Type != CharStreamConstants.EOF )
            {
                bool discard = false;
                // is there a channel override for token type?
                int channelI;
                if ( channelOverrideMap != null && channelOverrideMap.TryGetValue( t.Type, out channelI ) )
                    t.Channel = channelI;

                //if ( channelOverrideMap != null && channelOverrideMap.ContainsKey( t.getType() ) )
                //{
                //    object channelI = channelOverrideMap.get( t.getType() );
                //    if ( channelI != null )
                //    {
                //        t.setChannel( (int)channelI );
                //    }
                //}
                if ( discardSet != null &&
                     discardSet.Contains( t.Type ) )
                {
                    discard = true;
                }
                else if ( discardOffChannelTokens && t.Channel != this.channel )
                {
                    discard = true;
                }
                if ( !discard )
                {
                    t.TokenIndex = index;
                    tokens.Add( t );
                    index++;
                }
                t = tokenSource.NextToken();
            }
            // leave p pointing at first token on channel
            p = 0;
            p = SkipOffTokenChannels( p );
        }

        /** <summary>
         *  Move the input pointer to the next incoming token.  The stream
         *  must become active with LT(1) available.  consume() simply
         *  moves the input pointer so that LT(1) points at the next
         *  input symbol. Consume at least one token.
         *  </summary>
         *
         *  <remarks>
         *  Walk past any token not on the channel the parser is listening to.
         *  </remarks>
         */
        public virtual void Consume()
        {
            if ( p < tokens.Count )
            {
                p++;
                p = SkipOffTokenChannels( p ); // leave p on valid token
            }
        }

        /** <summary>Given a starting index, return the index of the first on-channel token.</summary> */
        protected virtual int SkipOffTokenChannels( int i )
        {
            int n = tokens.Count;
            while ( i < n && ( (IToken)tokens[i] ).Channel != channel )
            {
                i++;
            }
            return i;
        }

        protected virtual int SkipOffTokenChannelsReverse( int i )
        {
            while ( i >= 0 && ( (IToken)tokens[i] ).Channel != channel )
            {
                i--;
            }
            return i;
        }

        /** <summary>
         *  A simple filter mechanism whereby you can tell this token stream
         *  to force all tokens of type ttype to be on channel.  For example,
         *  when interpreting, we cannot exec actions so we need to tell
         *  the stream to force all WS and NEWLINE to be a different, ignored
         *  channel.
         *  </summary>
         */
        public virtual void SetTokenTypeChannel( int ttype, int channel )
        {
            if ( channelOverrideMap == null )
            {
                channelOverrideMap = new Dictionary<int, int>();
            }
            channelOverrideMap[ttype] = channel;
        }

        public virtual void DiscardTokenType( int ttype )
        {
            if ( discardSet == null )
            {
                discardSet = new HashSet<int>();
            }
            discardSet.Add( ttype );
        }

        public virtual void SetDiscardOffChannelTokens( bool discardOffChannelTokens )
        {
            this.discardOffChannelTokens = discardOffChannelTokens;
        }

        public virtual IList<IToken> GetTokens()
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            return tokens;
        }

        public virtual IList<IToken> GetTokens( int start, int stop )
        {
            return GetTokens( start, stop, (BitSet)null );
        }

        /** <summary>
         *  Given a start and stop index, return a List of all tokens in
         *  the token type BitSet.  Return null if no tokens were found.  This
         *  method looks at both on and off channel tokens.
         *  </summary>
         */
        public virtual IList<IToken> GetTokens( int start, int stop, BitSet types )
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            if ( stop >= tokens.Count )
            {
                stop = tokens.Count - 1;
            }
            if ( start < 0 )
            {
                start = 0;
            }
            if ( start > stop )
            {
                return null;
            }

            // list = tokens[start:stop]:{Token t, t.getType() in types}
            IList<IToken> filteredTokens = new List<IToken>();
            for ( int i = start; i <= stop; i++ )
            {
                IToken t = tokens[i];
                if ( types == null || types.Member( t.Type ) )
                {
                    filteredTokens.Add( t );
                }
            }
            if ( filteredTokens.Count == 0 )
            {
                filteredTokens = null;
            }
            return filteredTokens;
        }

        public virtual IList<IToken> GetTokens( int start, int stop, IList<int> types )
        {
            return GetTokens( start, stop, new BitSet( types ) );
        }

        public virtual IList<IToken> GetTokens( int start, int stop, int ttype )
        {
            return GetTokens( start, stop, BitSet.Of( ttype ) );
        }

        /** <summary>
         *  Get the ith token from the current position 1..n where k=1 is the
         *  first symbol of lookahead.
         *  </summary>
         */
        public virtual IToken LT( int k )
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            if ( k == 0 )
            {
                return null;
            }
            if ( k < 0 )
            {
                return LB( -k );
            }
            //System.out.print("LT(p="+p+","+k+")=");
            if ( ( p + k - 1 ) >= tokens.Count )
            {
                return TokenConstants.EOF_TOKEN;
            }
            //System.out.println(tokens.get(p+k-1));
            int i = p;
            int n = 1;
            // find k good tokens
            while ( n < k )
            {
                // skip off-channel tokens
                i = SkipOffTokenChannels( i + 1 ); // leave p on valid token
                n++;
            }
            if ( i >= tokens.Count )
            {
                return TokenConstants.EOF_TOKEN;
            }
            return (IToken)tokens[i];
        }

        /** <summary>Look backwards k tokens on-channel tokens</summary> */
        protected virtual IToken LB( int k )
        {
            //System.out.print("LB(p="+p+","+k+") ");
            if ( p == -1 )
            {
                FillBuffer();
            }
            if ( k == 0 )
            {
                return null;
            }
            if ( ( p - k ) < 0 )
            {
                return null;
            }

            int i = p;
            int n = 1;
            // find k good tokens looking backwards
            while ( n <= k )
            {
                // skip off-channel tokens
                i = SkipOffTokenChannelsReverse( i - 1 ); // leave p on valid token
                n++;
            }
            if ( i < 0 )
            {
                return null;
            }
            return (IToken)tokens[i];
        }

        /** <summary>
         *  Return absolute token i; ignore which channel the tokens are on;
         *  that is, count all tokens not just on-channel tokens.
         *  </summary>
         */
        public virtual IToken Get( int i )
        {
            return (IToken)tokens[i];
        }

        public virtual int LA( int i )
        {
            return LT( i ).Type;
        }

        public virtual int Mark()
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            lastMarker = Index;
            return lastMarker;
        }

        public virtual void Release( int marker )
        {
            // no resources to release
        }

        public virtual int Size()
        {
            return tokens.Count;
        }

        public virtual void Rewind( int marker )
        {
            Seek( marker );
        }

        public virtual void Rewind()
        {
            Seek( lastMarker );
        }

        public virtual void Reset()
        {
            p = 0;
            lastMarker = 0;
        }

        public virtual void Seek( int index )
        {
            p = index;
        }

        public virtual ITokenSource TokenSource
        {
            get
            {
                return tokenSource;
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
            if ( p == -1 )
            {
                FillBuffer();
            }
            return ToString( 0, tokens.Count - 1 );
        }

        public virtual string ToString( int start, int stop )
        {
            if ( start < 0 || stop < 0 )
            {
                return null;
            }
            if ( p == -1 )
            {
                FillBuffer();
            }
            if ( stop >= tokens.Count )
            {
                stop = tokens.Count - 1;
            }
            StringBuilder buf = new StringBuilder();
            for ( int i = start; i <= stop; i++ )
            {
                IToken t = tokens[i];
                buf.Append( t.Text );
            }
            return buf.ToString();
        }

        public virtual string ToString( IToken start, IToken stop )
        {
            if ( start != null && stop != null )
            {
                return ToString( start.TokenIndex, stop.TokenIndex );
            }
            return null;
        }
    }
}
