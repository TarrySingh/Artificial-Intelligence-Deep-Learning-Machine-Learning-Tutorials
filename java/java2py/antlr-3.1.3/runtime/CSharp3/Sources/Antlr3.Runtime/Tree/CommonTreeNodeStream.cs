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
    using Antlr.Runtime.Misc;

    using StringBuilder = System.Text.StringBuilder;

    [System.Serializable]
    public class CommonTreeNodeStream : LookaheadStream<object>, ITreeNodeStream
    {
        public const int DEFAULT_INITIAL_BUFFER_SIZE = 100;
        public const int INITIAL_CALL_STACK_SIZE = 10;

        /** <summary>Pull nodes from which tree?</summary> */
        object _root;

        /** <summary>If this tree (root) was created from a token stream, track it.</summary> */
        protected ITokenStream tokens;

        /** <summary>What tree adaptor was used to build these trees</summary> */
        [System.NonSerialized]
        ITreeAdaptor _adaptor;

        /** The tree iterator we are using */
        TreeIterator _it;

        /** <summary>Stack of indexes used for push/pop calls</summary> */
        Stack<int> _calls;

        /** <summary>Tree (nil A B C) trees like flat A B C streams</summary> */
        bool _hasNilRoot = false;

        /** <summary>Tracks tree depth.  Level=0 means we're at root node level.</summary> */
        int _level = 0;

        public CommonTreeNodeStream( object tree )
            : this( new CommonTreeAdaptor(), tree )
        {
        }

        public CommonTreeNodeStream( ITreeAdaptor adaptor, object tree )
            : base( adaptor.Create(TokenConstants.EOF,"EOF") ) // set EOF
        {
            this._root = tree;
            this._adaptor = adaptor;
            _it = new TreeIterator( adaptor, _root );
            _it.eof = this.Eof; // make sure tree iterator returns the EOF we want
        }

        #region Properties
        public virtual string SourceName
        {
            get
            {
                if ( TokenStream == null )
                    return null;

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
                return _adaptor;
            }
            set
            {
                _adaptor = value;
            }
        }
        public virtual object TreeSource
        {
            get
            {
                return _root;
            }
        }
        public virtual bool UniqueNavigationNodes
        {
            get
            {
                return false;
            }
            set
            {
            }
        }
        #endregion

        public virtual void Reset()
        {
            base.Clear();
            _it.Reset();
            _hasNilRoot = false;
            _level = 0;
            if ( _calls != null )
                _calls.Clear();
        }

        public override object NextElement()
        {
            _it.MoveNext();
            object t = _it.Current;
            //System.out.println("pulled "+adaptor.getType(t));
            if ( t == _it.up )
            {
                _level--;
                if ( _level == 0 && _hasNilRoot )
                {
                    _it.MoveNext();
                    return _it.Current; // don't give last UP; get EOF
                }
            }
            else if ( t == _it.down )
            {
                _level++;
            }

            if ( _level == 0 && _adaptor.IsNil( t ) )
            {
                // if nil root, scarf nil, DOWN
                _hasNilRoot = true;
                _it.MoveNext();
                t = _it.Current; // t is now DOWN, so get first real node next
                _level++;
                _it.MoveNext();
                t = _it.Current;
            }
            return t;
        }

        public virtual int LA( int i )
        {
            return _adaptor.GetType( LT( i ) );
        }

        /** Make stream jump to a new location, saving old location.
         *  Switch back with pop().
         */
        public virtual void Push( int index )
        {
            if ( _calls == null )
            {
                _calls = new Stack<int>();
            }
            _calls.Push( _p ); // save current index
            Seek( index );
        }

        /** Seek back to previous index saved during last push() call.
         *  Return top of stack (return index).
         */
        public virtual int Pop()
        {
            int ret = _calls.Pop();
            Seek( ret );
            return ret;
        }

        #region Tree rewrite interface

        public virtual void ReplaceChildren( object parent, int startChildIndex, int stopChildIndex, object t )
        {
            if ( parent != null )
            {
                _adaptor.ReplaceChildren( parent, startChildIndex, stopChildIndex, t );
            }
        }

        #endregion

        public virtual string ToString( object start, object stop )
        {
            // we'll have to walk from start to stop in tree; we're not keeping
            // a complete node stream buffer
            return "n/a";
        }

        /** <summary>For debugging; destructive: moves tree iterator to end.</summary> */
        public virtual string ToTokenTypeString()
        {
            Reset();
            StringBuilder buf = new StringBuilder();
            object o = LT( 1 );
            int type = _adaptor.GetType( o );
            while ( type != TokenConstants.EOF )
            {
                buf.Append( " " );
                buf.Append( type );
                Consume();
                o = LT( 1 );
                type = _adaptor.GetType( o );
            }
            return buf.ToString();
        }

        #region IIntStream Members

        int IIntStream.Size()
        {
            return Count;
        }

        #endregion
    }
}
