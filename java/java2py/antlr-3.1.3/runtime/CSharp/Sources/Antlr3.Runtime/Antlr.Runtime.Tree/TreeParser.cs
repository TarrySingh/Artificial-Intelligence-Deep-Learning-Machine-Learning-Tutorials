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


namespace Antlr.Runtime.Tree
{
	using System;
	using System.Text.RegularExpressions;
	using Antlr.Runtime;
	
	/// <summary>
	/// A parser for a stream of tree nodes.  "tree grammars" result in a subclass
	/// of this.  All the error reporting and recovery is shared with Parser via
	/// the BaseRecognizer superclass.
	/// </summary>
	public class TreeParser : BaseRecognizer
	{
		public const int DOWN = Token.DOWN;
		public const int UP   = Token.UP;

	    // precompiled regex used by InContext()
	    static readonly string dotdot = @".*[^.]\.\.[^.].*";
	    static readonly string doubleEtc = @".*\.\.\.\s+\.\.\..*";
	    static readonly string spaces = @"\s+";
	    static readonly Regex dotdotPattern = new Regex(dotdot, RegexOptions.Compiled);
	    static readonly Regex doubleEtcPattern = new Regex(doubleEtc, RegexOptions.Compiled);
	    static readonly Regex spacesPattern = new Regex(spaces, RegexOptions.Compiled);

		public TreeParser(ITreeNodeStream input)
			: base() // highlight that we go to super to set state object
		{
			TreeNodeStream = input;
		}

		public TreeParser(ITreeNodeStream input, RecognizerSharedState state)
			: base(state) // share the state object with another parser
		{
			TreeNodeStream = input;
		}

		/// <summary>Set the input stream</summary>
		virtual public ITreeNodeStream TreeNodeStream
		{
			get { return input; }
			set { this.input = value; }
		}

		override public string SourceName {
			get { return input.SourceName; }
		}

		protected override object GetCurrentInputSymbol(IIntStream input) {
			return ((ITreeNodeStream)input).LT(1);
		}

		protected override object GetMissingSymbol(IIntStream input,
										  RecognitionException e,
										  int expectedTokenType,
										  BitSet follow) {
			string tokenText = "<missing " + TokenNames[expectedTokenType] + ">";
			return new CommonTree(new CommonToken(expectedTokenType, tokenText));
		}	

		/// <summary>Reset the parser </summary>
		override public void Reset() 
		{
			base.Reset(); // reset all recognizer state variables
			if ( input != null ) 
			{
				input.Seek(0); // rewind the input
			}
		}

		/// <summary>
		/// Match '.' in tree parser.
		/// </summary>
		/// <remarks>
		/// Match '.' in tree parser has special meaning.  Skip node or
		/// entire tree if node has children.  If children, scan until
		/// corresponding UP node.
		/// </remarks>
		override public void MatchAny(IIntStream ignore /* ignore stream param, it's a copy of this.input */) 
		{
			state.errorRecovery = false;
			state.failed = false;
			object look = input.LT(1);
			if ( input.TreeAdaptor.GetChildCount(look) == 0 ) 
			{
				input.Consume(); // not subtree, consume 1 node and return
				return;
			}
			// current node is a subtree, skip to corresponding UP.
			// must count nesting level to get right UP
			int level = 0;
			int tokenType = input.TreeAdaptor.GetNodeType(look);
			while ( (tokenType != Token.EOF) && !( (tokenType == UP) && (level == 0) ) ) 
			{
				input.Consume();
				look = input.LT(1);
				tokenType = input.TreeAdaptor.GetNodeType(look);
				if ( tokenType == DOWN ) 
				{
					level++;
				}
				else if ( tokenType == UP ) 
				{
					level--;
				}
			}
			input.Consume(); // consume UP
		}

		override public IIntStream Input
		{
			get { return input; } 
		}

		protected internal ITreeNodeStream input;

		/// <summary>We have DOWN/UP nodes in the stream that have no line info; override.
		/// plus we want to alter the exception type. Don't try to recover
		/// from tree parser errors inline...
		/// </summary>
		protected internal override object RecoverFromMismatchedToken(IIntStream input, int ttype, BitSet follow) {
			throw new MismatchedTreeNodeException(ttype, (ITreeNodeStream)input);
		}

		/// <summary>
		/// Prefix error message with the grammar name because message is
		/// always intended for the programmer because the parser built
		/// the input tree not the user.
		/// </summary>
		override public string GetErrorHeader(RecognitionException e)
		{
			return GrammarFileName + ": node from "
				+ (e.approximateLineInfo ? "after " : "")
				+ "line " + e.Line + ":" + e.CharPositionInLine;
		}

		/// <summary>
		/// Tree parsers parse nodes they usually have a token object as
		/// payload. Set the exception token and do the default behavior.
		/// 
		/// </summary>
		override public string GetErrorMessage(RecognitionException e, string[] tokenNames) 
		{
			if ( this is TreeParser ) 
			{
				ITreeAdaptor adaptor = ((ITreeNodeStream)e.Input).TreeAdaptor;
				e.Token = adaptor.GetToken(e.Node);
				if ( e.Token == null ) 
				{ // could be an UP/DOWN node
					e.Token = new CommonToken(adaptor.GetNodeType(e.Node), adaptor.GetNodeText(e.Node));
				}
			}
			return base.GetErrorMessage(e, tokenNames);
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