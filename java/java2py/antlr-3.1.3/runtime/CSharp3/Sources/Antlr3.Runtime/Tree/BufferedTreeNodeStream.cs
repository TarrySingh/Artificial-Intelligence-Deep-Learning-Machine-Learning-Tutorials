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

namespace Antlr.Runtime.Tree
{
    using System.Collections.Generic;

    using Console = System.Console;
    using IList = System.Collections.IList;
    using StringBuilder = System.Text.StringBuilder;

    /** <summary>A buffered stream of tree nodes.  Nodes can be from a tree of ANY kind.</summary>
     *
     *  This node stream sucks all nodes out of the tree specified in
     *  the constructor during construction and makes pointers into
     *  the tree using an array of Object pointers. The stream necessarily
     *  includes pointers to DOWN and UP and EOF nodes.
     *
     *  This stream knows how to mark/release for backtracking.
     *
     *  This stream is most suitable for tree interpreters that need to
     *  jump around a lot or for tree parsers requiring speed (at cost of memory).
     *  There is some duplicated functionality here with UnBufferedTreeNodeStream
     *  but just in bookkeeping, not tree walking etc...
     *
     *  TARGET DEVELOPERS:
     *
     *  This is the old CommonTreeNodeStream that buffered up entire node stream.
     *  No need to implement really as new CommonTreeNodeStream is much better
     *  and covers what we need.
     *
     *  @see CommonTreeNodeStream
     */
    public class BufferedTreeNodeStream : ITreeNodeStream
    {
        public const int DEFAULT_INITIAL_BUFFER_SIZE = 100;
        public const int INITIAL_CALL_STACK_SIZE = 10;

        protected sealed class StreamIterator : IEnumerator<object>
        {
            BufferedTreeNodeStream _outer;
            int _index;

            public StreamIterator( BufferedTreeNodeStream outer )
            {
                _outer = outer;
                _index = -1;
            }

            #region IEnumerator<object> Members

            public object Current
            {
                get
                {
                    if ( _index < _outer.nodes.Count )
                        return _outer.nodes[_index];

                    return _outer.eof;
                }
            }

            #endregion

            #region IDisposable Members

            public void Dispose()
            {
            }

            #endregion

            #region IEnumerator Members

            public bool MoveNext()
            {
                if ( _index < _outer.nodes.Count )
                    _index++;

                return _index < _outer.nodes.Count;
            }

            public void Reset()
            {
                _index = -1;
            }

            #endregion
        }

        // all these navigation nodes are shared and hence they
        // cannot contain any line/column info

        protected object down;
        protected object up;
        protected object eof;

        /** <summary>The complete mapping from stream index to tree node.
         *  This buffer includes pointers to DOWN, UP, and EOF nodes.
         *  It is built upon ctor invocation.  The elements are type
         *  Object as we don't what the trees look like.</summary>
         *
         *  Load upon first need of the buffer so we can set token types
         *  of interest for reverseIndexing.  Slows us down a wee bit to
         *  do all of the if p==-1 testing everywhere though.
         */
        protected IList nodes;

        /** <summary>Pull nodes from which tree?</summary> */
        protected object root;

        /** <summary>IF this tree (root) was created from a token stream, track it.</summary> */
        protected ITokenStream tokens;

        /** <summary>What tree adaptor was used to build these trees</summary> */
        ITreeAdaptor adaptor;

        /** <summary>Reuse same DOWN, UP navigation nodes unless this is true</summary> */
        protected bool uniqueNavigationNodes = false;

        /** <summary>The index into the nodes list of the current node (next node
         *  to consume).  If -1, nodes array not filled yet.</summary>
         */
        protected int p = -1;

        /** <summary>Track the last mark() call result value for use in rewind().</summary> */
        protected int lastMarker;

        /** <summary>Stack of indexes used for push/pop calls</summary> */
        protected Stack<int> calls;

        public BufferedTreeNodeStream( object tree )
            : this( new CommonTreeAdaptor(), tree )
        {
        }

        public BufferedTreeNodeStream( ITreeAdaptor adaptor, object tree )
            : this( adaptor, tree, DEFAULT_INITIAL_BUFFER_SIZE )
        {
        }

        public BufferedTreeNodeStream( ITreeAdaptor adaptor, object tree, int initialBufferSize )
        {
            this.root = tree;
            this.adaptor = adaptor;
            nodes = new List<object>( initialBufferSize );
            down = adaptor.Create( TokenConstants.DOWN, "DOWN" );
            up = adaptor.Create( TokenConstants.UP, "UP" );
            eof = adaptor.Create( TokenConstants.EOF, "EOF" );
        }

        #region Properties

        public virtual object TreeSource
        {
            get
            {
                return root;
            }
        }

        public virtual string SourceName
        {
            get
            {
                return TokenStream.SourceName;
            }
        }

        public virtual ITokenStream TokenStream
        {
            get
            {
                return tokens;
            }
            set
            {
                tokens = value;
            }
        }

        public virtual ITreeAdaptor TreeAdaptor
        {
            get
            {
                return adaptor;
            }
            set
            {
                adaptor = value;
            }
        }

        public virtual bool UniqueNavigationNodes
        {
            get
            {
                return uniqueNavigationNodes;
            }
            set
            {
                uniqueNavigationNodes = value;
            }
        }

        #endregion

        /** Walk tree with depth-first-search and fill nodes buffer.
         *  Don't do DOWN, UP nodes if its a list (t is isNil).
         */
        protected virtual void FillBuffer()
        {
            FillBuffer( root );
            //Console.Out.WriteLine( "revIndex=" + tokenTypeToStreamIndexesMap );
            p = 0; // buffer of nodes intialized now
        }

        public virtual void FillBuffer( object t )
        {
            bool nil = adaptor.IsNil( t );
            if ( !nil )
            {
                nodes.Add( t ); // add this node
            }
            // add DOWN node if t has children
            int n = adaptor.GetChildCount( t );
            if ( !nil && n > 0 )
            {
                AddNavigationNode( TokenConstants.DOWN );
            }
            // and now add all its children
            for ( int c = 0; c < n; c++ )
            {
                object child = adaptor.GetChild( t, c );
                FillBuffer( child );
            }
            // add UP node if t has children
            if ( !nil && n > 0 )
            {
                AddNavigationNode( TokenConstants.UP );
            }
        }

        /** What is the stream index for node? 0..n-1
         *  Return -1 if node not found.
         */
        protected virtual int GetNodeIndex( object node )
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            for ( int i = 0; i < nodes.Count; i++ )
            {
                object t = nodes[i];
                if ( t == node )
                {
                    return i;
                }
            }
            return -1;
        }

        /** As we flatten the tree, we use UP, DOWN nodes to represent
         *  the tree structure.  When debugging we need unique nodes
         *  so instantiate new ones when uniqueNavigationNodes is true.
         */
        protected virtual void AddNavigationNode( int ttype )
        {
            object navNode = null;
            if ( ttype == TokenConstants.DOWN )
            {
                if ( UniqueNavigationNodes )
                {
                    navNode = adaptor.Create( TokenConstants.DOWN, "DOWN" );
                }
                else
                {
                    navNode = down;
                }
            }
            else
            {
                if ( UniqueNavigationNodes )
                {
                    navNode = adaptor.Create( TokenConstants.UP, "UP" );
                }
                else
                {
                    navNode = up;
                }
            }
            nodes.Add( navNode );
        }

        public virtual object this[int i]
        {
            get
            {
                if ( p == -1 )
                {
                    FillBuffer();
                }
                return nodes[i];
            }
        }

        public virtual object LT( int k )
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
            if ( ( p + k - 1 ) >= nodes.Count )
            {
                return eof;
            }
            return nodes[p + k - 1];
        }

        public virtual object GetCurrentSymbol()
        {
            return LT( 1 );
        }

#if false
        public virtual object getLastTreeNode()
        {
            int i = Index;
            if ( i >= size() )
            {
                i--; // if at EOF, have to start one back
            }
            Console.Out.WriteLine( "start last node: " + i + " size==" + nodes.Count );
            while ( i >= 0 &&
                ( adaptor.getType( this[i] ) == TokenConstants.EOF ||
                 adaptor.getType( this[i] ) == TokenConstants.UP ||
                 adaptor.getType( this[i] ) == TokenConstants.DOWN ) )
            {
                i--;
            }
            Console.Out.WriteLine( "stop at node: " + i + " " + nodes[i] );
            return nodes[i];
        }
#endif

        /** <summary>Look backwards k nodes</summary> */
        protected virtual object LB( int k )
        {
            if ( k == 0 )
            {
                return null;
            }
            if ( ( p - k ) < 0 )
            {
                return null;
            }
            return nodes[p - k];
        }

        public virtual void Consume()
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            p++;
        }

        public virtual int LA( int i )
        {
            return adaptor.GetType( LT( i ) );
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

        public virtual int Index
        {
            get
            {
                return p;
            }
        }

        public virtual void Rewind( int marker )
        {
            Seek( marker );
        }

        public virtual void Rewind()
        {
            Seek( lastMarker );
        }

        public virtual void Seek( int index )
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            p = index;
        }

        /** <summary>
         *  Make stream jump to a new location, saving old location.
         *  Switch back with pop().
         *  </summary>
         */
        public virtual void Push( int index )
        {
            if ( calls == null )
            {
                calls = new Stack<int>();
            }
            calls.Push( p ); // save current index
            Seek( index );
        }

        /** <summary>
         *  Seek back to previous index saved during last push() call.
         *  Return top of stack (return index).
         *  </summary>
         */
        public virtual int Pop()
        {
            int ret = calls.Pop();
            Seek( ret );
            return ret;
        }

        public virtual void Reset()
        {
            p = 0;
            lastMarker = 0;
            if ( calls != null )
            {
                calls.Clear();
            }
        }

        public virtual int Size()
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            return nodes.Count;
        }

        public virtual IEnumerator<object> Iterator()
        {
            if ( p == -1 )
            {
                FillBuffer();
            }

            return new StreamIterator( this );
        }

        // TREE REWRITE INTERFACE

        public virtual void ReplaceChildren( object parent, int startChildIndex, int stopChildIndex, object t )
        {
            if ( parent != null )
            {
                adaptor.ReplaceChildren( parent, startChildIndex, stopChildIndex, t );
            }
        }

        /** <summary>Used for testing, just return the token type stream</summary> */
        public virtual string ToTokenTypeString()
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            StringBuilder buf = new StringBuilder();
            for ( int i = 0; i < nodes.Count; i++ )
            {
                object t = nodes[i];
                buf.Append( " " );
                buf.Append( adaptor.GetType( t ) );
            }
            return buf.ToString();
        }

        /** <summary>Debugging</summary> */
        public virtual string ToTokenString( int start, int stop )
        {
            if ( p == -1 )
            {
                FillBuffer();
            }
            StringBuilder buf = new StringBuilder();
            for ( int i = start; i < nodes.Count && i <= stop; i++ )
            {
                object t = nodes[i];
                buf.Append( " " );
                buf.Append( adaptor.GetToken( t ) );
            }
            return buf.ToString();
        }

        public virtual string ToString( object start, object stop )
        {
            Console.Out.WriteLine( "toString" );
            if ( start == null || stop == null )
            {
                return null;
            }
            if ( p == -1 )
            {
                FillBuffer();
            }
            //Console.Out.WriteLine( "stop: " + stop );
            if ( start is CommonTree )
                Console.Out.Write( "toString: " + ( (CommonTree)start ).Token + ", " );
            else
                Console.Out.WriteLine( start );
            if ( stop is CommonTree )
                Console.Out.WriteLine( ( (CommonTree)stop ).Token );
            else
                Console.Out.WriteLine( stop );
            // if we have the token stream, use that to dump text in order
            if ( tokens != null )
            {
                int beginTokenIndex = adaptor.GetTokenStartIndex( start );
                int endTokenIndex = adaptor.GetTokenStopIndex( stop );
                // if it's a tree, use start/stop index from start node
                // else use token range from start/stop nodes
                if ( adaptor.GetType( stop ) == TokenConstants.UP )
                {
                    endTokenIndex = adaptor.GetTokenStopIndex( start );
                }
                else if ( adaptor.GetType( stop ) == TokenConstants.EOF )
                {
                    endTokenIndex = Size() - 2; // don't use EOF
                }
                return tokens.ToString( beginTokenIndex, endTokenIndex );
            }
            // walk nodes looking for start
            object t = null;
            int i = 0;
            for ( ; i < nodes.Count; i++ )
            {
                t = nodes[i];
                if ( t == start )
                {
                    break;
                }
            }
            // now walk until we see stop, filling string buffer with text
            StringBuilder buf = new StringBuilder();
            t = nodes[i];
            while ( t != stop )
            {
                string text = adaptor.GetText( t );
                if ( text == null )
                {
                    text = " " + adaptor.GetType( t ).ToString();
                }
                buf.Append( text );
                i++;
                t = nodes[i];
            }
            // include stop node too
            string text2 = adaptor.GetText( stop );
            if ( text2 == null )
            {
                text2 = " " + adaptor.GetType( stop ).ToString();
            }
            buf.Append( text2 );
            return buf.ToString();
        }
    }
}
