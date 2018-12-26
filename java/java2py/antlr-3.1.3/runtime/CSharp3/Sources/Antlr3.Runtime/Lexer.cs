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
     *  A lexer is recognizer that draws input symbols from a character stream.
     *  lexer grammars result in a subclass of this object. A Lexer object
     *  uses simplified match() and error recovery mechanisms in the interest
     *  of speed.
     *  </summary>
     */
    public abstract class Lexer : BaseRecognizer, ITokenSource
    {
        /** <summary>Where is the lexer drawing characters from?</summary> */
        protected ICharStream input;

        public Lexer()
        {
        }

        public Lexer( ICharStream input )
        {
            this.input = input;
        }

        public Lexer( ICharStream input, RecognizerSharedState state ) :
            base( state )
        {
            this.input = input;
        }

        #region Properties
        public string Text
        {
            get
            {
                return GetText();
            }
            set
            {
                SetText( value );
            }
        }
        public int Line
        {
            get
            {
                return input.Line;
            }
            set
            {
                input.Line = value;
            }
        }
        public int CharPositionInLine
        {
            get
            {
                return input.CharPositionInLine;
            }
            set
            {
                input.CharPositionInLine = value;
            }
        }
        public string[] TokenNames
        {
            get
            {
                return GetTokenNames();
            }
        }
        #endregion

        public override void Reset()
        {
            base.Reset(); // reset all recognizer state variables
            // wack Lexer state variables
            if ( input != null )
            {
                input.Seek( 0 ); // rewind the input
            }
            if ( state == null )
            {
                return; // no shared state work to do
            }
            state.token = null;
            state.type = TokenConstants.INVALID_TOKEN_TYPE;
            state.channel = TokenConstants.DEFAULT_CHANNEL;
            state.tokenStartCharIndex = -1;
            state.tokenStartCharPositionInLine = -1;
            state.tokenStartLine = -1;
            state.text = null;
        }

        /** <summary>Return a token from this source; i.e., match a token on the char stream.</summary> */
        public virtual IToken NextToken()
        {
            for ( ; ; )
            {
                state.token = null;
                state.channel = TokenConstants.DEFAULT_CHANNEL;
                state.tokenStartCharIndex = input.Index;
                state.tokenStartCharPositionInLine = input.CharPositionInLine;
                state.tokenStartLine = input.Line;
                state.text = null;
                if ( input.LA( 1 ) == CharStreamConstants.EOF )
                {
                    return TokenConstants.EOF_TOKEN;
                }
                try
                {
                    mTokens();
                    if ( state.token == null )
                    {
                        Emit();
                    }
                    else if ( state.token == TokenConstants.SKIP_TOKEN )
                    {
                        continue;
                    }
                    return state.token;
                }
                catch ( NoViableAltException nva )
                {
                    ReportError( nva );
                    Recover( nva ); // throw out current char and try again
                }
                catch ( RecognitionException re )
                {
                    ReportError( re );
                    // match() routine has already called recover()
                }
            }
        }

        /** <summary>
         *  Instruct the lexer to skip creating a token for current lexer rule
         *  and look for another token.  nextToken() knows to keep looking when
         *  a lexer rule finishes with token set to SKIP_TOKEN.  Recall that
         *  if token==null at end of any token rule, it creates one for you
         *  and emits it.
         *  </summary>
         */
        public virtual void Skip()
        {
            state.token = TokenConstants.SKIP_TOKEN;
        }

        /** <summary>This is the lexer entry point that sets instance var 'token'</summary> */
        public abstract void mTokens();

        /** <summary>Set the char stream and reset the lexer</summary> */
        public virtual void SetCharStream( ICharStream input )
        {
            this.input = null;
            Reset();
            this.input = input;
        }

        public virtual ICharStream GetCharStream()
        {
            return this.input;
        }

        public override string SourceName
        {
            get
            {
                return input.SourceName;
            }
        }

        /** <summary>
         *  Currently does not support multiple emits per nextToken invocation
         *  for efficiency reasons.  Subclass and override this method and
         *  nextToken (to push tokens into a list and pull from that list rather
         *  than a single variable as this implementation does).
         *  </summary>
         */
        public virtual void Emit( IToken token )
        {
            state.token = token;
        }

        /** <summary>
         *  The standard method called to automatically emit a token at the
         *  outermost lexical rule.  The token object should point into the
         *  char buffer start..stop.  If there is a text override in 'text',
         *  use that to set the token's text.  Override this method to emit
         *  custom Token objects.
         *  </summary>
         *
         *  <remarks>
         *  If you are building trees, then you should also override
         *  Parser or TreeParser.getMissingSymbol().
         *  </remarks>
         */
        public virtual IToken Emit()
        {
            IToken t = new CommonToken( input, state.type, state.channel, state.tokenStartCharIndex, GetCharIndex() - 1 );
            t.Line = state.tokenStartLine;
            t.Text = state.text;
            t.CharPositionInLine = state.tokenStartCharPositionInLine;
            Emit( t );
            return t;
        }

        public virtual void Match( string s )
        {
            int i = 0;
            while ( i < s.Length )
            {
                if ( input.LA( 1 ) != s[i] )
                {
                    if ( state.backtracking > 0 )
                    {
                        state.failed = true;
                        return;
                    }
                    MismatchedTokenException mte =
                        new MismatchedTokenException( s[i], input )
                        {
                            tokenNames = GetTokenNames()
                        };
                    Recover( mte );
                    throw mte;
                }
                i++;
                input.Consume();
                state.failed = false;
            }
        }

        public virtual void MatchAny()
        {
            input.Consume();
        }

        public virtual void Match( int c )
        {
            if ( input.LA( 1 ) != c )
            {
                if ( state.backtracking > 0 )
                {
                    state.failed = true;
                    return;
                }
                MismatchedTokenException mte =
                    new MismatchedTokenException( c, input )
                    {
                        tokenNames = GetTokenNames()
                    };
                Recover( mte );  // don't really recover; just consume in lexer
                throw mte;
            }
            input.Consume();
            state.failed = false;
        }

        public virtual void MatchRange( int a, int b )
        {
            if ( input.LA( 1 ) < a || input.LA( 1 ) > b )
            {
                if ( state.backtracking > 0 )
                {
                    state.failed = true;
                    return;
                }
                MismatchedRangeException mre =
                    new MismatchedRangeException( a, b, input );
                Recover( mre );
                throw mre;
            }
            input.Consume();
            state.failed = false;
        }

        /** <summary>What is the index of the current character of lookahead?</summary> */
        public virtual int GetCharIndex()
        {
            return input.Index;
        }

        /** <summary>Return the text matched so far for the current token or any text override.</summary> */
        public virtual string GetText()
        {
            if ( state.text != null )
            {
                return state.text;
            }
            return input.substring( state.tokenStartCharIndex, GetCharIndex() - 1 );
        }

        /** <summary>Set the complete text of this token; it wipes any previous changes to the text.</summary> */
        public virtual void SetText( string text )
        {
            state.text = text;
        }

        public override void ReportError( RecognitionException e )
        {
            /** TODO: not thought about recovery in lexer yet.
             *
            // if we've already reported an error and have not matched a token
            // yet successfully, don't report any errors.
            if ( errorRecovery ) {
                //System.err.print("[SPURIOUS] ");
                return;
            }
            errorRecovery = true;
             */

            DisplayRecognitionError( this.GetTokenNames(), e );
        }

        public override string GetErrorMessage( RecognitionException e, string[] tokenNames )
        {
            string msg = null;
            if ( e is MismatchedTokenException )
            {
                MismatchedTokenException mte = (MismatchedTokenException)e;
                msg = "mismatched character " + GetCharErrorDisplay( e.c ) + " expecting " + GetCharErrorDisplay( mte.expecting );
            }
            else if ( e is NoViableAltException )
            {
                NoViableAltException nvae = (NoViableAltException)e;
                // for development, can add "decision=<<"+nvae.grammarDecisionDescription+">>"
                // and "(decision="+nvae.decisionNumber+") and
                // "state "+nvae.stateNumber
                msg = "no viable alternative at character " + GetCharErrorDisplay( e.c );
            }
            else if ( e is EarlyExitException )
            {
                EarlyExitException eee = (EarlyExitException)e;
                // for development, can add "(decision="+eee.decisionNumber+")"
                msg = "required (...)+ loop did not match anything at character " + GetCharErrorDisplay( e.c );
            }
            else if ( e is MismatchedNotSetException )
            {
                MismatchedNotSetException mse = (MismatchedNotSetException)e;
                msg = "mismatched character " + GetCharErrorDisplay( e.c ) + " expecting set " + mse.expecting;
            }
            else if ( e is MismatchedSetException )
            {
                MismatchedSetException mse = (MismatchedSetException)e;
                msg = "mismatched character " + GetCharErrorDisplay( e.c ) + " expecting set " + mse.expecting;
            }
            else if ( e is MismatchedRangeException )
            {
                MismatchedRangeException mre = (MismatchedRangeException)e;
                msg = "mismatched character " + GetCharErrorDisplay( e.c ) + " expecting set " +
                      GetCharErrorDisplay( mre.a ) + ".." + GetCharErrorDisplay( mre.b );
            }
            else
            {
                msg = base.GetErrorMessage( e, tokenNames );
            }
            return msg;
        }

        public virtual string GetCharErrorDisplay( int c )
        {
            string s = c.ToString(); //string.valueOf((char)c);
            switch ( c )
            {
            case TokenConstants.EOF:
                s = "<EOF>";
                break;
            case '\n':
                s = "\\n";
                break;
            case '\t':
                s = "\\t";
                break;
            case '\r':
                s = "\\r";
                break;
            }
            return "'" + s + "'";
        }

        /** <summary>
         *  Lexers can normally match any char in it's vocabulary after matching
         *  a token, so do the easy thing and just kill a character and hope
         *  it all works out.  You can instead use the rule invocation stack
         *  to do sophisticated error recovery if you are in a fragment rule.
         *  </summary>
         */
        public virtual void Recover( RecognitionException re )
        {
            //System.out.println("consuming char "+(char)input.LA(1)+" during recovery");
            //re.printStackTrace();
            input.Consume();
        }

        public virtual void TraceIn( string ruleName, int ruleIndex )
        {
            string inputSymbol = ( (char)input.LT( 1 ) ) + " line=" + Line + ":" + CharPositionInLine;
            base.TraceIn( ruleName, ruleIndex, inputSymbol );
        }

        public virtual void TraceOut( string ruleName, int ruleIndex )
        {
            string inputSymbol = ( (char)input.LT( 1 ) ) + " line=" + Line + ":" + CharPositionInLine;
            base.TraceOut( ruleName, ruleIndex, inputSymbol );
        }
    }
}
