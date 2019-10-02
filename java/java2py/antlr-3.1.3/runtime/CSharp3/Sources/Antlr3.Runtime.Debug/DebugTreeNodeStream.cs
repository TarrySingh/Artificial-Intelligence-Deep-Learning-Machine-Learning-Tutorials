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
    using Obsolete = System.ObsoleteAttribute;
    using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;
    using ITreeNodeStream = Antlr.Runtime.Tree.ITreeNodeStream;

    /** <summary>
     *  Debug any tree node stream.  The constructor accepts the stream
     *  and a debug listener.  As node stream calls come in, debug events
     *  are triggered.
     *  </summary>
     */
    public class DebugTreeNodeStream : ITreeNodeStream
    {
        protected IDebugEventListener dbg;
        protected ITreeAdaptor adaptor;
        protected ITreeNodeStream input;
        protected bool initialStreamState = true;

        /** <summary>Track the last mark() call result value for use in rewind().</summary> */
        protected int lastMarker;

        public DebugTreeNodeStream( ITreeNodeStream input,
                                   IDebugEventListener dbg )
        {
            this.input = input;
            this.adaptor = input.TreeAdaptor;
            this.input.UniqueNavigationNodes = true;
            DebugListener = dbg;
        }

        #region Properties
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
        public virtual int Index
        {
            get
            {
                return input.Index;
            }
        }
        public virtual ITokenStream TokenStream
        {
            get
            {
                return input.TokenStream;
            }
        }
        public virtual ITreeAdaptor TreeAdaptor
        {
            get
            {
                return adaptor;
            }
        }
        public virtual object TreeSource
        {
            get
            {
                return input;
            }
        }
        /** <summary>
         *  It is normally this object that instructs the node stream to
         *  create unique nav nodes, but to satisfy interface, we have to
         *  define it.  It might be better to ignore the parameter but
         *  there might be a use for it later, so I'll leave.
         *  </summary>
         */
        public bool UniqueNavigationNodes
        {
            get
            {
                return input.UniqueNavigationNodes;
            }
            set
            {
                input.UniqueNavigationNodes = value;
            }
        }

        #endregion

        [Obsolete]
        public void SetDebugListener( IDebugEventListener dbg )
        {
            DebugListener = dbg;
        }

        [Obsolete]
        public ITreeAdaptor GetTreeAdaptor()
        {
            return TreeAdaptor;
        }

        public virtual void Consume()
        {
            object node = input.LT( 1 );
            input.Consume();
            dbg.ConsumeNode( node );
        }

        public virtual object this[int i]
        {
            get
            {
                return input[i];
            }
        }

        public virtual object LT( int i )
        {
            object node = input.LT( i );
            int ID = adaptor.GetUniqueID( node );
            string text = adaptor.GetText( node );
            int type = adaptor.GetType( node );
            dbg.LT( i, node );
            return node;
        }

        public virtual int LA( int i )
        {
            object node = input.LT( i );
            int ID = adaptor.GetUniqueID( node );
            string text = adaptor.GetText( node );
            int type = adaptor.GetType( node );
            dbg.LT( i, node );
            return type;
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

        [Obsolete]
        public object GetTreeSource()
        {
            return TreeSource;
        }

        public virtual string SourceName
        {
            get
            {
                return TokenStream.SourceName;
            }
        }

        [Obsolete]
        public ITokenStream GetTokenStream()
        {
            return TokenStream;
        }

        public virtual void ReplaceChildren( object parent, int startChildIndex, int stopChildIndex, object t )
        {
            input.ReplaceChildren( parent, startChildIndex, stopChildIndex, t );
        }

        public virtual string ToString( object start, object stop )
        {
            return input.ToString( start, stop );
        }
    }
}
