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
    using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;

    /** <summary>
     *  A TreeAdaptor proxy that fires debugging events to a DebugEventListener
     *  delegate and uses the TreeAdaptor delegate to do the actual work.  All
     *  AST events are triggered by this adaptor; no code gen changes are needed
     *  in generated rules.  Debugging events are triggered *after* invoking
     *  tree adaptor routines.
     *  </summary>
     *
     *  <remarks>
     *  Trees created with actions in rewrite actions like "-&gt; ^(ADD {foo} {bar})"
     *  cannot be tracked as they might not use the adaptor to create foo, bar.
     *  The debug listener has to deal with tree node IDs for which it did
     *  not see a createNode event.  A single &lt;unknown&gt; node is sufficient even
     *  if it represents a whole tree.
     *  </remarks>
     */
    public class DebugTreeAdaptor : ITreeAdaptor
    {
        protected IDebugEventListener dbg;
        protected ITreeAdaptor adaptor;

        public DebugTreeAdaptor( IDebugEventListener dbg, ITreeAdaptor adaptor )
        {
            this.dbg = dbg;
            this.adaptor = adaptor;
        }

        public virtual object Create( IToken payload )
        {
            if ( payload.TokenIndex < 0 )
            {
                // could be token conjured up during error recovery
                return Create( payload.Type, payload.Text );
            }
            object node = adaptor.Create( payload );
            dbg.CreateNode( node, payload );
            return node;
        }

        public virtual object ErrorNode( ITokenStream input, IToken start, IToken stop,
                                RecognitionException e )
        {
            object node = adaptor.ErrorNode( input, start, stop, e );
            if ( node != null )
            {
                dbg.ErrorNode( node );
            }
            return node;
        }

        public virtual object DupTree( object tree )
        {
            object t = adaptor.DupTree( tree );
            // walk the tree and emit create and add child events
            // to simulate what dupTree has done. dupTree does not call this debug
            // adapter so I must simulate.
            SimulateTreeConstruction( t );
            return t;
        }

        /** <summary>^(A B C): emit create A, create B, add child, ...</summary> */
        protected virtual void SimulateTreeConstruction( object t )
        {
            dbg.CreateNode( t );
            int n = adaptor.GetChildCount( t );
            for ( int i = 0; i < n; i++ )
            {
                object child = adaptor.GetChild( t, i );
                SimulateTreeConstruction( child );
                dbg.AddChild( t, child );
            }
        }

        public virtual object DupNode( object treeNode )
        {
            object d = adaptor.DupNode( treeNode );
            dbg.CreateNode( d );
            return d;
        }

        public virtual object Nil()
        {
            object node = adaptor.Nil();
            dbg.NilNode( node );
            return node;
        }

        public virtual bool IsNil( object tree )
        {
            return adaptor.IsNil( tree );
        }

        public virtual void AddChild( object t, object child )
        {
            if ( t == null || child == null )
            {
                return;
            }
            adaptor.AddChild( t, child );
            dbg.AddChild( t, child );
        }

        public virtual object BecomeRoot( object newRoot, object oldRoot )
        {
            object n = adaptor.BecomeRoot( newRoot, oldRoot );
            dbg.BecomeRoot( newRoot, oldRoot );
            return n;
        }

        public virtual object RulePostProcessing( object root )
        {
            return adaptor.RulePostProcessing( root );
        }

        public virtual void AddChild( object t, IToken child )
        {
            object n = this.Create( child );
            this.AddChild( t, n );
        }

        public virtual object BecomeRoot( IToken newRoot, object oldRoot )
        {
            object n = this.Create( newRoot );
            adaptor.BecomeRoot( n, oldRoot );
            dbg.BecomeRoot( newRoot, oldRoot );
            return n;
        }

        public virtual object Create( int tokenType, IToken fromToken )
        {
            object node = adaptor.Create( tokenType, fromToken );
            dbg.CreateNode( node );
            return node;
        }

        public virtual object Create( int tokenType, IToken fromToken, string text )
        {
            object node = adaptor.Create( tokenType, fromToken, text );
            dbg.CreateNode( node );
            return node;
        }

        public virtual object Create( int tokenType, string text )
        {
            object node = adaptor.Create( tokenType, text );
            dbg.CreateNode( node );
            return node;
        }

        public virtual int GetType( object t )
        {
            return adaptor.GetType( t );
        }

        public virtual void SetType( object t, int type )
        {
            adaptor.SetType( t, type );
        }

        public virtual string GetText( object t )
        {
            return adaptor.GetText( t );
        }

        public virtual void SetText( object t, string text )
        {
            adaptor.SetText( t, text );
        }

        public virtual IToken GetToken( object t )
        {
            return adaptor.GetToken( t );
        }

        public virtual void SetTokenBoundaries( object t, IToken startToken, IToken stopToken )
        {
            adaptor.SetTokenBoundaries( t, startToken, stopToken );
            if ( t != null && startToken != null && stopToken != null )
            {
                dbg.SetTokenBoundaries(
                    t, startToken.TokenIndex,
                    stopToken.TokenIndex );
            }
        }

        public virtual int GetTokenStartIndex( object t )
        {
            return adaptor.GetTokenStartIndex( t );
        }

        public virtual int GetTokenStopIndex( object t )
        {
            return adaptor.GetTokenStopIndex( t );
        }

        public virtual object GetChild( object t, int i )
        {
            return adaptor.GetChild( t, i );
        }

        public virtual void SetChild( object t, int i, object child )
        {
            adaptor.SetChild( t, i, child );
        }

        public virtual object DeleteChild( object t, int i )
        {
            return DeleteChild( t, i );
        }

        public virtual int GetChildCount( object t )
        {
            return adaptor.GetChildCount( t );
        }

        public virtual int GetUniqueID( object node )
        {
            return adaptor.GetUniqueID( node );
        }

        public virtual object GetParent( object t )
        {
            return adaptor.GetParent( t );
        }

        public virtual int GetChildIndex( object t )
        {
            return adaptor.GetChildIndex( t );
        }

        public virtual void SetParent( object t, object parent )
        {
            adaptor.SetParent( t, parent );
        }

        public virtual void SetChildIndex( object t, int index )
        {
            adaptor.SetChildIndex( t, index );
        }

        public virtual void ReplaceChildren( object parent, int startChildIndex, int stopChildIndex, object t )
        {
            adaptor.ReplaceChildren( parent, startChildIndex, stopChildIndex, t );
        }

        #region support

        public virtual IDebugEventListener GetDebugListener()
        {
            return dbg;
        }

        public virtual void SetDebugListener( IDebugEventListener dbg )
        {
            this.dbg = dbg;
        }

        public virtual ITreeAdaptor GetTreeAdaptor()
        {
            return adaptor;
        }

        #endregion
    }
}
