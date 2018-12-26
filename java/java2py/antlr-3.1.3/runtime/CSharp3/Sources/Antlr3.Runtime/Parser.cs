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

namespace Antlr.Runtime
{
    /** <summary>
     *  A parser for TokenStreams.  "parser grammars" result in a subclass
     *  of this.
     *  </summary>
     */
    public class Parser : BaseRecognizer
    {
        public ITokenStream input;

        public Parser( ITokenStream input )
            : base()
        {
            //super(); // highlight that we go to super to set state object
            TokenStream = input;
        }

        public Parser( ITokenStream input, RecognizerSharedState state )
            : base( state )
        {
            //super(state); // share the state object with another parser
            TokenStream = input;
        }

        public override void Reset()
        {
            base.Reset(); // reset all recognizer state variables
            if ( input != null )
            {
                input.Seek( 0 ); // rewind the input
            }
        }

        protected override object GetCurrentInputSymbol( IIntStream input )
        {
            return ( (ITokenStream)input ).LT( 1 );
        }

        protected override object GetMissingSymbol( IIntStream input,
                                          RecognitionException e,
                                          int expectedTokenType,
                                          BitSet follow )
        {
            string tokenText = null;
            if ( expectedTokenType == TokenConstants.EOF )
                tokenText = "<missing EOF>";
            else
                tokenText = "<missing " + GetTokenNames()[expectedTokenType] + ">";
            CommonToken t = new CommonToken( expectedTokenType, tokenText );
            IToken current = ( (ITokenStream)input ).LT( 1 );
            if ( current.Type == TokenConstants.EOF )
            {
                current = ( (ITokenStream)input ).LT( -1 );
            }
            t.Line = current.Line;
            t.CharPositionInLine = current.CharPositionInLine;
            t.Channel = DEFAULT_TOKEN_CHANNEL;
            return t;
        }

        /** <summary>Gets or sets the token stream; resets the parser upon a set.</summary> */
        public virtual ITokenStream TokenStream
        {
            get
            {
                return input;
            }
            set
            {
                input = null;
                Reset();
                input = value;
            }
        }

        public override string SourceName
        {
            get
            {
                return input.SourceName;
            }
        }

        public virtual void TraceIn( string ruleName, int ruleIndex )
        {
            base.TraceIn( ruleName, ruleIndex, input.LT( 1 ) );
        }

        public virtual void TraceOut( string ruleName, int ruleIndex )
        {
            base.TraceOut( ruleName, ruleIndex, input.LT( 1 ) );
        }

#if false
        protected bool EvaluatePredicate( System.Action predicate )
        {
            state.backtracking++;
            int start = input.Mark();
            try
            {
                predicate();
            }
            catch ( RecognitionException re )
            {
                System.Console.Error.WriteLine( "impossible: " + re );
            }
            bool success = !state.failed;
            input.Rewind( start );
            state.backtracking--;
            state.failed = false;
            return success;
        }
#endif
    }
}
