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
    using System.Collections.Generic;

    /** <summary>
     *  Broadcast debug events to multiple listeners.  Lets you debug and still
     *  use the event mechanism to build parse trees etc...  Not thread-safe.
     *  Don't add events in one thread while parser fires events in another.
     *  </summary>
     *
     *  <seealso cref="DebugEventRepeater"/>
     */
    public class DebugEventHub : IDebugEventListener
    {
        protected IList<IDebugEventListener> _listeners = new List<IDebugEventListener>();

        public DebugEventHub( params IDebugEventListener[] listeners )
        {
            _listeners = new List<IDebugEventListener>( listeners );
        }

        public virtual void Initialize()
        {
        }

        /** <summary>
         *  Add another listener to broadcast events too.  Not thread-safe.
         *  Don't add events in one thread while parser fires events in another.
         *  </summary>
         */
        public virtual void AddListener( IDebugEventListener listener )
        {
            _listeners.Add( listener );
        }

        /* To avoid a mess like this:
            public void enterRule(final String ruleName) {
                broadcast(new Code(){
                    public void exec(DebugEventListener listener) {listener.enterRule(ruleName);}}
                    );
            }
            I am dup'ing the for-loop in each.  Where are Java closures!? blech!
         */

        public virtual void EnterRule( string grammarFileName, string ruleName )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.EnterRule( grammarFileName, ruleName );
            }
        }

        public virtual void ExitRule( string grammarFileName, string ruleName )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.ExitRule( grammarFileName, ruleName );
            }
        }

        public virtual void EnterAlt( int alt )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.EnterAlt( alt );
            }
        }

        public virtual void EnterSubRule( int decisionNumber )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.EnterSubRule( decisionNumber );
            }
        }

        public virtual void ExitSubRule( int decisionNumber )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.ExitSubRule( decisionNumber );
            }
        }

        public virtual void EnterDecision( int decisionNumber )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.EnterDecision( decisionNumber );
            }
        }

        public virtual void ExitDecision( int decisionNumber )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.ExitDecision( decisionNumber );
            }
        }

        public virtual void Location( int line, int pos )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.Location( line, pos );
            }
        }

        public virtual void ConsumeToken( IToken token )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.ConsumeToken( token );
            }
        }

        public virtual void ConsumeHiddenToken( IToken token )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.ConsumeHiddenToken( token );
            }
        }

        public virtual void LT( int index, IToken t )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.LT( index, t );
            }
        }

        public virtual void Mark( int index )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.Mark( index );
            }
        }

        public virtual void Rewind( int index )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.Rewind( index );
            }
        }

        public virtual void Rewind()
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.Rewind();
            }
        }

        public virtual void BeginBacktrack( int level )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.BeginBacktrack( level );
            }
        }

        public virtual void EndBacktrack( int level, bool successful )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.EndBacktrack( level, successful );
            }
        }

        public virtual void RecognitionException( RecognitionException e )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.RecognitionException( e );
            }
        }

        public virtual void BeginResync()
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.BeginResync();
            }
        }

        public virtual void EndResync()
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.EndResync();
            }
        }

        public virtual void SemanticPredicate( bool result, string predicate )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.SemanticPredicate( result, predicate );
            }
        }

        public virtual void Commence()
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.Commence();
            }
        }

        public virtual void Terminate()
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.Terminate();
            }
        }


        #region Tree parsing stuff

        public virtual void ConsumeNode( object t )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.ConsumeNode( t );
            }
        }

        public virtual void LT( int index, object t )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.LT( index, t );
            }
        }

        #endregion


        #region AST Stuff

        public virtual void NilNode( object t )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.NilNode( t );
            }
        }

        public virtual void ErrorNode( object t )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.ErrorNode( t );
            }
        }

        public virtual void CreateNode( object t )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.CreateNode( t );
            }
        }

        public virtual void CreateNode( object node, IToken token )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.CreateNode( node, token );
            }
        }

        public virtual void BecomeRoot( object newRoot, object oldRoot )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.BecomeRoot( newRoot, oldRoot );
            }
        }

        public virtual void AddChild( object root, object child )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.AddChild( root, child );
            }
        }

        public virtual void SetTokenBoundaries( object t, int tokenStartIndex, int tokenStopIndex )
        {
            for ( int i = 0; i < _listeners.Count; i++ )
            {
                IDebugEventListener listener = _listeners[i];
                listener.SetTokenBoundaries( t, tokenStartIndex, tokenStopIndex );
            }
        }

        #endregion
    }
}
