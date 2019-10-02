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
     *  A blank listener that does nothing; useful for real classes so
     *  they don't have to have lots of blank methods and are less
     *  sensitive to updates to debug interface.
     *  </summary>
     */
    public class BlankDebugEventListener : IDebugEventListener
    {
        public int RuleLevel
        {
            get;
            protected set;
        }

        public virtual void Initialize()
        {
        }

        public virtual void EnterRule( string grammarFileName, string ruleName )
        {
            if ( RuleLevel == 0 )
                Commence();
            RuleLevel++;
        }
        public virtual void ExitRule( string grammarFileName, string ruleName )
        {
            RuleLevel--;
            if ( RuleLevel == 0 )
                Terminate();
        }
        public virtual void EnterAlt( int alt )
        {
        }
        public virtual void EnterSubRule( int decisionNumber )
        {
        }
        public virtual void ExitSubRule( int decisionNumber )
        {
        }
        public virtual void EnterDecision( int decisionNumber )
        {
        }
        public virtual void ExitDecision( int decisionNumber )
        {
        }
        public virtual void Location( int line, int pos )
        {
        }
        public virtual void ConsumeToken( IToken token )
        {
        }
        public virtual void ConsumeHiddenToken( IToken token )
        {
        }
        public virtual void LT( int i, IToken t )
        {
        }
        public virtual void Mark( int i )
        {
        }
        public virtual void Rewind( int i )
        {
        }
        public virtual void Rewind()
        {
        }
        public virtual void BeginBacktrack( int level )
        {
        }
        public virtual void EndBacktrack( int level, bool successful )
        {
        }
        public virtual void RecognitionException( RecognitionException e )
        {
        }
        public virtual void BeginResync()
        {
        }
        public virtual void EndResync()
        {
        }
        public virtual void SemanticPredicate( bool result, string predicate )
        {
        }
        public virtual void Commence()
        {
        }
        public virtual void Terminate()
        {
        }

        #region Tree parsing stuff

        public virtual void ConsumeNode( object t )
        {
        }
        public virtual void LT( int i, object t )
        {
        }

        #endregion


        #region AST Stuff

        public virtual void NilNode( object t )
        {
        }
        public virtual void ErrorNode( object t )
        {
        }
        public virtual void CreateNode( object t )
        {
        }
        public virtual void CreateNode( object node, IToken token )
        {
        }
        public virtual void BecomeRoot( object newRoot, object oldRoot )
        {
        }
        public virtual void AddChild( object root, object child )
        {
        }
        public virtual void SetTokenBoundaries( object t, int tokenStartIndex, int tokenStopIndex )
        {
        }

        #endregion
    }
}

