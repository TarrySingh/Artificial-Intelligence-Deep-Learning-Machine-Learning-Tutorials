/*
[The "BSD licence"]
Copyright (c) 2007-2008 Johannes Luber
Copyright (c) 2005-2007 Kunle Odutola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


namespace Antlr.Runtime
{
	using System;
	using CollectionUtils = Antlr.Runtime.Collections.CollectionUtils;
	
	/// <summary>
	/// A lexer is recognizer that draws input symbols from a character stream.
	/// lexer grammars result in a subclass of this object. A Lexer object
	/// uses simplified Match() and error recovery mechanisms in the interest
	/// of speed.
	/// </summary>
	public abstract class Lexer : BaseRecognizer, ITokenSource
	{
		const int TOKEN_dot_EOF = (int)CharStreamConstants.EOF;

		#region Constructors

		public Lexer()
		{
		}

		public Lexer(ICharStream input)
		{
			this.input = input;
		}

		public Lexer(ICharStream input, RecognizerSharedState state)
			: base(state) {
			this.input = input;
		}
	
		#endregion

		#region Public API

		/// <summary>Set the char stream and reset the lexer </summary>
		virtual public ICharStream CharStream
		{
			get { return this.input;  }
			set
			{
				this.input = null;
				Reset();
				this.input = value;
			}
		}

		override public string SourceName {
			get { return input.SourceName; }
		}


		override public IIntStream Input
		{
			get { return input; }
		}

		virtual public int Line
		{
			get { return input.Line; }
		}

		virtual public int CharPositionInLine
		{
			get { return input.CharPositionInLine; }
		}

		/// <summary>What is the index of the current character of lookahead? </summary>
		virtual public int CharIndex
		{
			get { return input.Index(); }
		}

		/// <summary>
		/// Gets or sets the 'lexeme' for the current token.
		/// </summary>
		/// <remarks>
		/// <para>
		/// The getter returns the text matched so far for the current token or any 
		/// text override.
		/// </para>
		/// <para>
		/// The setter sets the complete text of this token. It overrides/wipes any 
		/// previous changes to the text.
		/// </para>
		/// </remarks>
		virtual public string Text
		{
			get 
			{
				if (state.text != null)
				{
					return state.text;
				}
				return input.Substring(state.tokenStartCharIndex, CharIndex - 1); 
			}
			set
			{
				state.text = value;
			}
		}

		override public void Reset() 
		{
			base.Reset(); // reset all recognizer state variables
			// wack Lexer state variables
			if (input != null) {
				input.Seek(0); // rewind the input
			}
			if (state == null) {
				return; // no shared state work to do
			}
			state.token = null;
			state.type = Token.INVALID_TOKEN_TYPE;
			state.channel = Token.DEFAULT_CHANNEL;
			state.tokenStartCharIndex = -1;
			state.tokenStartCharPositionInLine = -1;
			state.tokenStartLine = -1;
			state.text = null;
		}

		/// <summary>
		/// Return a token from this source; i.e., Match a token on the char stream.
		/// </summary>
		public virtual IToken NextToken()
		{
			while (true)
			{
				state.token = null;
				state.channel = Token.DEFAULT_CHANNEL;
				state.tokenStartCharIndex = input.Index();
				state.tokenStartCharPositionInLine = input.CharPositionInLine;
				state.tokenStartLine = input.Line;
				state.text = null;
				if (input.LA(1) == (int)CharStreamConstants.EOF)
				{
					return Token.EOF_TOKEN;
				}
				try
				{
					mTokens();
					if (state.token == null)
					{
						Emit();
					}
					else if (state.token == Token.SKIP_TOKEN)
					{
						continue;
					}
					return state.token;
				}
				catch (NoViableAltException nva) {
					ReportError(nva);
					Recover(nva); // throw out current char and try again
				}
				catch (RecognitionException re) {
					ReportError(re);
					// Match() routine has already called Recover()
				}
			}
		}

		/// <summary>
		/// Instruct the lexer to skip creating a token for current lexer rule and 
		/// look for another token.  NextToken() knows to keep looking when a lexer 
		/// rule finishes with token set to SKIP_TOKEN.  Recall that if token==null 
		/// at end of any token rule, it creates one for you and emits it. 
		/// </summary>
		public void Skip()
		{
			state.token = Token.SKIP_TOKEN;
		}

		/// <summary>This is the lexer entry point that sets instance var 'token' </summary>
		public abstract void mTokens();

		/// <summary>
		/// Currently does not support multiple emits per nextToken invocation
		/// for efficiency reasons.  Subclass and override this method and
		/// nextToken (to push tokens into a list and pull from that list rather
		/// than a single variable as this implementation does).
		/// </summary>
		public virtual void Emit(IToken token)
		{
			state.token = token;
		}

		/// <summary>
		/// The standard method called to automatically emit a token at the 
		/// outermost lexical rule.  The token object should point into the 
		/// char buffer start..stop.  If there is a text override in 'text', 
		/// use that to set the token's text. 
		/// </summary>
		/// <remarks><para>Override this method to emit custom Token objects.</para>
		/// <para>If you are building trees, then you should also override
		/// Parser or TreeParser.getMissingSymbol().</para>
		///</remarks>
		public virtual IToken Emit()
		{
			IToken t = new CommonToken(input, state.type, state.channel, state.tokenStartCharIndex, CharIndex - 1);
			t.Line = state.tokenStartLine;
			t.Text = state.text;
			t.CharPositionInLine = state.tokenStartCharPositionInLine;
			Emit(t);
			return t;
		}

		public virtual void Match(string s)
		{
			int i = 0;
			while (i < s.Length)
			{
				if (input.LA(1) != s[i])
				{
					if (state.backtracking > 0)
					{
						state.failed = true;
						return;
					}
					MismatchedTokenException mte = new MismatchedTokenException(s[i], input);
					Recover(mte); // don't really recover; just consume in lexer
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

		public virtual void Match(int c)
		{
			if (input.LA(1) != c)
			{
				if (state.backtracking > 0)
				{
					state.failed = true;
					return;
				}
				MismatchedTokenException mte = new MismatchedTokenException(c, input);
				Recover(mte);
				throw mte;
			}
			input.Consume();
			state.failed = false;
		}

		public virtual void MatchRange(int a, int b)
		{
			if (input.LA(1) < a || input.LA(1) > b)
			{
				if (state.backtracking > 0)
				{
					state.failed = true;
					return;
				}
				MismatchedRangeException mre = new MismatchedRangeException(a, b, input);
				Recover(mre);
				throw mre;
			}
			input.Consume();
			state.failed = false;
		}

		/// <summary>
		/// Lexers can normally Match any char in it's vocabulary after matching
		/// a token, so do the easy thing and just kill a character and hope
		/// it all works out.  You can instead use the rule invocation stack
		/// to do sophisticated error recovery if you are in a Fragment rule.
		/// </summary>
		public virtual void Recover(RecognitionException re)
		{
			input.Consume();
		}

		public override void ReportError(RecognitionException e)
		{
			DisplayRecognitionError(this.TokenNames, e);
		}

		override public string GetErrorMessage(RecognitionException e, string[] tokenNames)
		{
			string msg = null;

			if (e is MismatchedTokenException)
			{
				MismatchedTokenException mte = (MismatchedTokenException)e;
				msg = "mismatched character " + GetCharErrorDisplay(e.Char) + " expecting " + GetCharErrorDisplay(mte.Expecting);
			}
			else if (e is NoViableAltException)
			{
				NoViableAltException nvae = (NoViableAltException)e;
				// for development, can add "decision=<<"+nvae.grammarDecisionDescription+">>"
				// and "(decision="+nvae.decisionNumber+") and
				// "state "+nvae.stateNumber
				msg = "no viable alternative at character " + GetCharErrorDisplay(nvae.Char);
			}
			else if (e is EarlyExitException)
			{
				EarlyExitException eee = (EarlyExitException)e;
				// for development, can add "(decision="+eee.decisionNumber+")"
				msg = "required (...)+ loop did not match anything at character " + GetCharErrorDisplay(eee.Char);
			}
			else if (e is MismatchedNotSetException)
			{
				MismatchedSetException mse = (MismatchedSetException)e;
				msg = "mismatched character " + GetCharErrorDisplay(mse.Char) + " expecting set " + mse.expecting;
			}
			else if (e is MismatchedSetException)
			{
				MismatchedSetException mse = (MismatchedSetException)e;
				msg = "mismatched character " + GetCharErrorDisplay(mse.Char) + " expecting set " + mse.expecting;
			}
			else if (e is MismatchedRangeException)
			{
				MismatchedRangeException mre = (MismatchedRangeException)e;
				msg = "mismatched character " + GetCharErrorDisplay(mre.Char) + " expecting set " + GetCharErrorDisplay(mre.A) + ".." + GetCharErrorDisplay(mre.B);
			}
			else 
			{
				msg = base.GetErrorMessage(e, tokenNames);
			}
			return msg;
		}

		public string GetCharErrorDisplay(int c) 
		{
			string s;
			switch ( c ) 
			{
				//case Token.EOF :
				case TOKEN_dot_EOF :
					s = "<EOF>";
					break;
				case '\n' :
					s = "\\n";
					break;
				case '\t' :
					s = "\\t";
					break;
				case '\r' :
					s = "\\r";
					break;
				default:
					s = Convert.ToString((char)c);
					break;
			}
			return "'" + s + "'";
		}

		public virtual void TraceIn(string ruleName, int ruleIndex)  
		{
			string inputSymbol = ((char)input.LT(1)) + " line=" + Line + ":" + CharPositionInLine;
			base.TraceIn(ruleName, ruleIndex, inputSymbol);
		}

		public virtual void TraceOut(string ruleName, int ruleIndex)  
		{
			string inputSymbol = ((char)input.LT(1)) + " line=" + Line + ":" + CharPositionInLine;
			base.TraceOut(ruleName, ruleIndex, inputSymbol);
		}
		#endregion

		#region Data Members

		/// <summary>Where is the lexer drawing characters from? </summary>
		protected internal ICharStream input;
		
		#endregion
	}
}