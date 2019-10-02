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
    using Antlr.Runtime.JavaExtensions;

    using ParseTree = Antlr.Runtime.Tree.ParseTree;

    /** <summary>
     *  This parser listener tracks rule entry/exit and token matches
     *  to build a simple parse tree using ParseTree nodes.
     *  </summary>
     */
    public class ParseTreeBuilder : BlankDebugEventListener
    {
        public const string EPSILON_PAYLOAD = "<epsilon>";

        Stack<ParseTree> callStack = new Stack<ParseTree>();
        List<IToken> hiddenTokens = new List<IToken>();
        int backtracking = 0;

        public ParseTreeBuilder( string grammarName )
        {
            ParseTree root = Create( "<grammar " + grammarName + ">" );
            callStack.Push( root );
        }

        public virtual ParseTree GetTree()
        {
            var enumerator = callStack.GetEnumerator();
            enumerator.MoveNext();
            return enumerator.Current;
        }

        /** <summary>
         *  What kind of node to create.  You might want to override
         *  so I factored out creation here.
         *  </summary>
         */
        public virtual ParseTree Create( object payload )
        {
            return new ParseTree( payload );
        }

        public virtual ParseTree EpsilonNode()
        {
            return Create( EPSILON_PAYLOAD );
        }

        /** <summary>Backtracking or cyclic DFA, don't want to add nodes to tree</summary> */
        public override void EnterDecision( int d )
        {
            backtracking++;
        }
        public override void ExitDecision( int i )
        {
            backtracking--;
        }

        public override void EnterRule( string filename, string ruleName )
        {
            if ( backtracking > 0 )
                return;
            ParseTree parentRuleNode = callStack.Peek();
            ParseTree ruleNode = Create( ruleName );
            parentRuleNode.AddChild( ruleNode );
            callStack.Push( ruleNode );
        }

        public override void ExitRule( string filename, string ruleName )
        {
            if ( backtracking > 0 )
                return;
            ParseTree ruleNode = callStack.Peek();
            if ( ruleNode.ChildCount == 0 )
            {
                ruleNode.AddChild( EpsilonNode() );
            }
            callStack.Pop();
        }

        public override void ConsumeToken( IToken token )
        {
            if ( backtracking > 0 )
                return;
            ParseTree ruleNode = callStack.Peek();
            ParseTree elementNode = Create( token );
            elementNode.hiddenTokens = this.hiddenTokens;
            this.hiddenTokens = new List<IToken>();
            ruleNode.AddChild( elementNode );
        }

        public override void ConsumeHiddenToken( IToken token )
        {
            if ( backtracking > 0 )
                return;
            hiddenTokens.Add( token );
        }

        public override void RecognitionException( RecognitionException e )
        {
            if ( backtracking > 0 )
                return;
            ParseTree ruleNode = callStack.Peek();
            ParseTree errorNode = Create( e );
            ruleNode.AddChild( errorNode );
        }
    }
}
