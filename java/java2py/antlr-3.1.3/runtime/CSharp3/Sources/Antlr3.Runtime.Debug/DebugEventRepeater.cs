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

    /** <summary>
     *  A simple event repeater (proxy) that delegates all functionality to the
     *  listener sent into the ctor.  Useful if you want to listen in on a few
     *  debug events w/o interrupting the debugger.  Just subclass the repeater
     *  and override the methods you want to listen in on.  Remember to call
     *  the method in this class so the event will continue on to the original
     *  recipient.
     *  </summary>
     *
     *  <seealso cref="DebugEventHub"/>
     */
    public class DebugEventRepeater : IDebugEventListener
    {
        protected IDebugEventListener _listener;

        public DebugEventRepeater( IDebugEventListener listener )
        {
            _listener = listener;
        }

        public virtual void Initialize()
        {
        }

        public virtual void EnterRule( string grammarFileName, string ruleName )
        {
            _listener.EnterRule( grammarFileName, ruleName );
        }
        public virtual void ExitRule( string grammarFileName, string ruleName )
        {
            _listener.ExitRule( grammarFileName, ruleName );
        }
        public virtual void EnterAlt( int alt )
        {
            _listener.EnterAlt( alt );
        }
        public virtual void EnterSubRule( int decisionNumber )
        {
            _listener.EnterSubRule( decisionNumber );
        }
        public virtual void ExitSubRule( int decisionNumber )
        {
            _listener.ExitSubRule( decisionNumber );
        }
        public virtual void EnterDecision( int decisionNumber )
        {
            _listener.EnterDecision( decisionNumber );
        }
        public virtual void ExitDecision( int decisionNumber )
        {
            _listener.ExitDecision( decisionNumber );
        }
        public virtual void Location( int line, int pos )
        {
            _listener.Location( line, pos );
        }
        public virtual void ConsumeToken( IToken token )
        {
            _listener.ConsumeToken( token );
        }
        public virtual void ConsumeHiddenToken( IToken token )
        {
            _listener.ConsumeHiddenToken( token );
        }
        public virtual void LT( int i, IToken t )
        {
            _listener.LT( i, t );
        }
        public virtual void Mark( int i )
        {
            _listener.Mark( i );
        }
        public virtual void Rewind( int i )
        {
            _listener.Rewind( i );
        }
        public virtual void Rewind()
        {
            _listener.Rewind();
        }
        public virtual void BeginBacktrack( int level )
        {
            _listener.BeginBacktrack( level );
        }
        public virtual void EndBacktrack( int level, bool successful )
        {
            _listener.EndBacktrack( level, successful );
        }
        public virtual void RecognitionException( RecognitionException e )
        {
            _listener.RecognitionException( e );
        }
        public virtual void BeginResync()
        {
            _listener.BeginResync();
        }
        public virtual void EndResync()
        {
            _listener.EndResync();
        }
        public virtual void SemanticPredicate( bool result, string predicate )
        {
            _listener.SemanticPredicate( result, predicate );
        }
        public virtual void Commence()
        {
            _listener.Commence();
        }
        public virtual void Terminate()
        {
            _listener.Terminate();
        }

        #region Tree parsing stuff

        public virtual void ConsumeNode( object t )
        {
            _listener.ConsumeNode( t );
        }
        public virtual void LT( int i, object t )
        {
            _listener.LT( i, t );
        }

        #endregion


        #region AST Stuff

        public virtual void NilNode( object t )
        {
            _listener.NilNode( t );
        }
        public virtual void ErrorNode( object t )
        {
            _listener.ErrorNode( t );
        }
        public virtual void CreateNode( object t )
        {
            _listener.CreateNode( t );
        }
        public virtual void CreateNode( object node, IToken token )
        {
            _listener.CreateNode( node, token );
        }
        public virtual void BecomeRoot( object newRoot, object oldRoot )
        {
            _listener.BecomeRoot( newRoot, oldRoot );
        }
        public virtual void AddChild( object root, object child )
        {
            _listener.AddChild( root, child );
        }
        public virtual void SetTokenBoundaries( object t, int tokenStartIndex, int tokenStopIndex )
        {
            _listener.SetTokenBoundaries( t, tokenStartIndex, tokenStopIndex );
        }

        #endregion
    }
}
