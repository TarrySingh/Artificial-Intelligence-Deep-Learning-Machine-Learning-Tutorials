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
    using Antlr.Runtime.JavaExtensions;

    using Console = System.Console;
    using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;

    /** <summary>Print out (most of) the events... Useful for debugging, testing...</summary> */
    public class TraceDebugEventListener : BlankDebugEventListener
    {
        ITreeAdaptor adaptor;

        public TraceDebugEventListener( ITreeAdaptor adaptor )
        {
            this.adaptor = adaptor;
        }

        public void EnterRule( string ruleName )
        {
            Console.Out.WriteLine( "enterRule " + ruleName );
        }
        public void ExitRule( string ruleName )
        {
            Console.Out.WriteLine( "exitRule " + ruleName );
        }
        public override void EnterSubRule( int decisionNumber )
        {
            Console.Out.WriteLine( "enterSubRule" );
        }
        public override void ExitSubRule( int decisionNumber )
        {
            Console.Out.WriteLine( "exitSubRule" );
        }
        public override void Location( int line, int pos )
        {
            Console.Out.WriteLine( "location " + line + ":" + pos );
        }

        #region Tree parsing stuff

        public override void ConsumeNode( object t )
        {
            int ID = adaptor.GetUniqueID( t );
            string text = adaptor.GetText( t );
            int type = adaptor.GetType( t );
            Console.Out.WriteLine( "consumeNode " + ID + " " + text + " " + type );
        }

        public override void LT( int i, object t )
        {
            int ID = adaptor.GetUniqueID( t );
            string text = adaptor.GetText( t );
            int type = adaptor.GetType( t );
            Console.Out.WriteLine( "LT " + i + " " + ID + " " + text + " " + type );
        }

        #endregion


        #region AST stuff

        public override void NilNode( object t )
        {
            Console.Out.WriteLine( "nilNode " + adaptor.GetUniqueID( t ) );
        }

        public override void CreateNode( object t )
        {
            int ID = adaptor.GetUniqueID( t );
            string text = adaptor.GetText( t );
            int type = adaptor.GetType( t );
            Console.Out.WriteLine( "create " + ID + ": " + text + ", " + type );
        }

        public override void CreateNode( object node, IToken token )
        {
            int ID = adaptor.GetUniqueID( node );
            string text = adaptor.GetText( node );
            int tokenIndex = token.TokenIndex;
            Console.Out.WriteLine( "create " + ID + ": " + tokenIndex );
        }

        public override void BecomeRoot( object newRoot, object oldRoot )
        {
            Console.Out.WriteLine( "becomeRoot " + adaptor.GetUniqueID( newRoot ) + ", " +
                               adaptor.GetUniqueID( oldRoot ) );
        }

        public override void AddChild( object root, object child )
        {
            Console.Out.WriteLine( "addChild " + adaptor.GetUniqueID( root ) + ", " +
                               adaptor.GetUniqueID( child ) );
        }

        public override void SetTokenBoundaries( object t, int tokenStartIndex, int tokenStopIndex )
        {
            Console.Out.WriteLine( "setTokenBoundaries " + adaptor.GetUniqueID( t ) + ", " +
                               tokenStartIndex + ", " + tokenStopIndex );
        }

        #endregion
    }
}
