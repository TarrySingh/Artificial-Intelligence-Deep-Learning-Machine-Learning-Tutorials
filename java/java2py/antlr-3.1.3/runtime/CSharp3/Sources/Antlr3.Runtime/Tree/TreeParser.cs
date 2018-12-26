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
    using Regex = System.Text.RegularExpressions.Regex;
    using RegexOptions = System.Text.RegularExpressions.RegexOptions;

    /** <summary>
     *  A parser for a stream of tree nodes.  "tree grammars" result in a subclass
     *  of this.  All the error reporting and recovery is shared with Parser via
     *  the BaseRecognizer superclass.
     *  </summary>
    */
    public class TreeParser : BaseRecognizer
    {
        public const int DOWN = TokenConstants.DOWN;
        public const int UP = TokenConstants.UP;

        // precompiled regex used by inContext
        static string dotdot = ".*[^.]\\.\\.[^.].*";
        static string doubleEtc = ".*\\.\\.\\.\\s+\\.\\.\\..*";
        static Regex dotdotPattern = new Regex( dotdot, RegexOptions.Compiled );
        static Regex doubleEtcPattern = new Regex( doubleEtc, RegexOptions.Compiled );

        protected ITreeNodeStream input;

        public TreeParser( ITreeNodeStream input )
            : base() // highlight that we go to super to set state object
        {
            SetTreeNodeStream( input );
        }

        public TreeParser( ITreeNodeStream input, RecognizerSharedState state )
            : base( state ) // share the state object with another parser
        {
            SetTreeNodeStream( input );
        }

        public override void Reset()
        {
            base.Reset(); // reset all recognizer state variables
            if ( input != null )
            {
                input.Seek( 0 ); // rewind the input
            }
        }

        /** <summary>Set the input stream</summary> */
        public virtual void SetTreeNodeStream( ITreeNodeStream input )
        {
            this.input = input;
        }

        public virtual ITreeNodeStream GetTreeNodeStream()
        {
            return input;
        }

        public override string SourceName
        {
            get
            {
                return input.SourceName;
            }
        }

        protected override object GetCurrentInputSymbol( IIntStream input )
        {
            return ( (ITreeNodeStream)input ).LT( 1 );
        }

        protected override object GetMissingSymbol( IIntStream input,
                                          RecognitionException e,
                                          int expectedTokenType,
                                          BitSet follow )
        {
            string tokenText =
                "<missing " + GetTokenNames()[expectedTokenType] + ">";
            return new CommonTree( new CommonToken( expectedTokenType, tokenText ) );
        }

        /** <summary>
         *  Match '.' in tree parser has special meaning.  Skip node or
         *  entire tree if node has children.  If children, scan until
         *  corresponding UP node.
         *  </summary>
         */
        public override void MatchAny( IIntStream ignore )
        {
            state.errorRecovery = false;
            state.failed = false;
            // always consume the current node
            input.Consume();
            // if the next node is DOWN, then the current node is a subtree:
            // skip to corresponding UP. must count nesting level to get right UP
            int look = input.LA( 1 );
            if ( look == DOWN )
            {
                input.Consume();
                int level = 1;
                while ( level > 0 )
                {
                    switch ( input.LA( 1 ) )
                    {
                    case DOWN:
                        level++;
                        break;
                    case UP:
                        level--;
                        break;
                    case TokenConstants.EOF:
                        return;
                    default:
                        break;
                    }
                    input.Consume();
                }
            }
        }

        /** <summary>
         *  We have DOWN/UP nodes in the stream that have no line info; override.
         *  plus we want to alter the exception type.  Don't try to recover
         *  from tree parser errors inline...
         *  </summary>
         */
        protected override object RecoverFromMismatchedToken( IIntStream input, int ttype, BitSet follow )
        {
            throw new MismatchedTreeNodeException( ttype, (ITreeNodeStream)input );
        }

        /** <summary>
         *  Prefix error message with the grammar name because message is
         *  always intended for the programmer because the parser built
         *  the input tree not the user.
         *  </summary>
         */
        public override string GetErrorHeader( RecognitionException e )
        {
            return GrammarFileName + ": node from " +
                   ( e.approximateLineInfo ? "after " : "" ) + "line " + e.line + ":" + e.charPositionInLine;
        }

        /** <summary>
         *  Tree parsers parse nodes they usually have a token object as
         *  payload. Set the exception token and do the default behavior.
         *  </summary>
         */
        public override string GetErrorMessage( RecognitionException e, string[] tokenNames )
        {
            if ( this is TreeParser )
            {
                ITreeAdaptor adaptor = ( (ITreeNodeStream)e.input ).TreeAdaptor;
                e.token = adaptor.GetToken( e.node );
                if ( e.token == null )
                { // could be an UP/DOWN node
                    e.token = new CommonToken( adaptor.GetType( e.node ),
                                              adaptor.GetText( e.node ) );
                }
            }
            return base.GetErrorMessage( e, tokenNames );
        }

        public virtual void TraceIn( string ruleName, int ruleIndex )
        {
            base.TraceIn( ruleName, ruleIndex, input.LT( 1 ) );
        }

        public virtual void TraceOut( string ruleName, int ruleIndex )
        {
            base.TraceOut( ruleName, ruleIndex, input.LT( 1 ) );
        }

    }
}
