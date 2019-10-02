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
	
	/// <summary>A parser for TokenStreams.  Parser grammars result in a subclass
	/// of this.
	/// </summary>
	public class Parser : BaseRecognizer
	{
		public Parser(ITokenStream input)
			: base() // highlight that we go to base class to set state object
		{
			TokenStream = input;
		}

		public Parser(ITokenStream input, RecognizerSharedState state)
			: base(state) // share the state object with another parser
		{
			TokenStream = input;
		}

		override public void Reset() 
		{
			base.Reset(); // reset all recognizer state variables
			if ( input != null ) 
			{
				input.Seek(0); // rewind the input
			}
		}

		protected override object GetCurrentInputSymbol(IIntStream input) {
			return ((ITokenStream)input).LT(1);
		}

		protected override object GetMissingSymbol(IIntStream input,
										  RecognitionException e,
										  int expectedTokenType,
										  BitSet follow)
		{
			String tokenText = null;
			if ( expectedTokenType==Token.EOF ) tokenText = "<missing EOF>";
			else tokenText = "<missing " + TokenNames[expectedTokenType] + ">";
			CommonToken t = new CommonToken(expectedTokenType, tokenText);
			IToken current = ((ITokenStream)input).LT(1);
			if (current.Type == Token.EOF) {
				current = ((ITokenStream)input).LT(-1);
			}
			t.line = current.Line;
			t.CharPositionInLine = current.CharPositionInLine;
			t.Channel = DEFAULT_TOKEN_CHANNEL;
			return t;
		}

		/// <summary>Set the token stream and reset the parser </summary>
		virtual public ITokenStream TokenStream
		{
			get { return input; }
			
			set
			{
				this.input = null;
				Reset();
				this.input = value;
			}
			
		}

		public override string SourceName {
			get { return input.SourceName; }
		}
	
		protected internal ITokenStream input;

		override public IIntStream Input
		{
			get { return input; }
		}

		public virtual void TraceIn(string ruleName, int ruleIndex)  
		{
			base.TraceIn(ruleName, ruleIndex, input.LT(1));
		}

		public virtual void TraceOut(string ruleName, int ruleIndex)  
		{
			base.TraceOut(ruleName, ruleIndex, input.LT(1));
		}

	}
}