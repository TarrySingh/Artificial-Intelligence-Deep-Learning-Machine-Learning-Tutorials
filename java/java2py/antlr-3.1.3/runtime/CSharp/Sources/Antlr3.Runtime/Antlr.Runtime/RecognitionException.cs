/*
[The "BSD licence"]
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
	using ITree = Antlr.Runtime.Tree.ITree;
	using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;
	using ITreeNodeStream = Antlr.Runtime.Tree.ITreeNodeStream;
	using CommonTreeNodeStream = Antlr.Runtime.Tree.CommonTreeNodeStream;
	using CommonTree = Antlr.Runtime.Tree.CommonTree;

	/// <summary>The root of the ANTLR exception hierarchy.</summary>
	/// <remarks>
	/// To avoid English-only error messages and to generally make things
	/// as flexible as possible, these exceptions are not created with strings,
	/// but rather the information necessary to generate an error.  Then
	/// the various reporting methods in Parser and Lexer can be overridden
	/// to generate a localized error message.  For example, MismatchedToken
	/// exceptions are built with the expected token type.
	/// So, don't expect getMessage() to return anything.
	/// 
	/// You can access the stack trace, which means that you can compute the 
	/// complete trace of rules from the start symbol. This gives you considerable 
	/// context information with which to generate useful error messages.
	/// 
	/// ANTLR generates code that throws exceptions upon recognition error and
	/// also generates code to catch these exceptions in each rule.  If you
	/// want to quit upon first error, you can turn off the automatic error
	/// handling mechanism using rulecatch action, but you still need to
	/// override methods mismatch and recoverFromMismatchSet.
	/// 
	/// In general, the recognition exceptions can track where in a grammar a
	/// problem occurred and/or what was the expected input.  While the parser
	/// knows its state (such as current input symbol and line info) that
	/// state can change before the exception is reported so current token index
	/// is computed and stored at exception time.  From this info, you can
	/// perhaps print an entire line of input not just a single token, for example.
	/// Better to just say the recognizer had a problem and then let the parser
	/// figure out a fancy report.
	/// </remarks>
	[Serializable]
	public class RecognitionException : Exception
	{
		#region Constructors

		/// <summary>Used for remote debugger deserialization </summary>
		public RecognitionException()
			: this(null, null, null)
		{
		}

		public RecognitionException(string message)
			: this(message, null, null)
		{
		}

		public RecognitionException(string message, Exception inner)
			: this(message, inner, null)
		{
		}

		public RecognitionException(IIntStream input)
			: this(null, null, input)
		{
		}

		public RecognitionException(string message, IIntStream input)
			: this(message, null, input)
		{
		}

		public RecognitionException(string message, Exception inner, IIntStream input)
			: base(message, inner)
		{
			this.input = input;
			this.index = input.Index();
			if (input is ITokenStream)
			{
				this.token = ((ITokenStream)input).LT(1);
				this.line = token.Line;
				this.charPositionInLine = token.CharPositionInLine;
			}
			if (input is ITreeNodeStream)
			{
				ExtractInformationFromTreeNodeStream(input);
			}
			else if (input is ICharStream)
			{
				this.c = input.LA(1);
				this.line = ((ICharStream)input).Line;
				this.charPositionInLine = ((ICharStream)input).CharPositionInLine;
			}
			else
			{
				this.c = input.LA(1);
			}
		}

		#endregion

		#region Public API

		/// <summary>Returns the input stream in which the error occurred</summary>
		public IIntStream Input
		{
			get { return input; }
			set { input = value; }
		}

		/// <summary>
		/// Returns the token/char index in the stream when the error occurred
		/// </summary>
		public int Index
		{
			get { return index; }
			set { index = value; }
		}

		/// <summary>
		/// Returns the current Token when the error occurred (for parsers 
		/// although a tree parser might also set the token)
		/// </summary>
		public IToken Token
		{
			get { return token; }
			set { token = value; }
		}

		/// <summary>
		/// Returns the [tree parser] node where the error occured (for tree parsers).
		/// </summary>
		public object Node
		{
			get { return node; }
			set { node = value; }
		}

		/// <summary>
		/// Returns the current char when the error occurred (for lexers)
		/// </summary>
		public int Char
		{
			get { return c; }
			set { c = value; }
		}

		/// <summary>
		/// Returns the character position in the line when the error 
		/// occurred (for lexers)
		/// </summary>
		public int CharPositionInLine
		{
			get { return charPositionInLine; }
			set { charPositionInLine = value; }
		}

		/// <summary>
		/// Returns the line at which the error occurred (for lexers)
		/// </summary>
		public int Line
		{
			get { return line; }
			set { line = value; }
		}

		/// <summary>
		/// Returns the token type or char of the unexpected input element
		/// </summary>
		virtual public int UnexpectedType
		{
			get
			{
				if (input is ITokenStream)
				{
					return token.Type;
				}
				else if (input is ITreeNodeStream)
				{
					ITreeNodeStream nodes = (ITreeNodeStream)input;
					ITreeAdaptor adaptor = nodes.TreeAdaptor;
					return adaptor.GetNodeType(node);
				}
				else
				{
					return c;
				}
			}

		}

		#endregion

		#region Non-Public API

		protected void ExtractInformationFromTreeNodeStream(IIntStream input)
		{
			ITreeNodeStream nodes = (ITreeNodeStream)input;
			this.node = nodes.LT(1);
			ITreeAdaptor adaptor = nodes.TreeAdaptor;
			IToken payload = adaptor.GetToken(node);
			if ( payload != null )
			{
				this.token = payload;
				if ( payload.Line <= 0 )
				{
					// imaginary node; no line/pos info; scan backwards
					int i = -1;
					object priorNode = nodes.LT(i);
					while ( priorNode != null )
					{
						IToken priorPayload = adaptor.GetToken(priorNode);
						if ( (priorPayload != null) && (priorPayload.Line > 0) )
						{
							// we found the most recent real line / pos info
							this.line = priorPayload.Line;
							this.charPositionInLine = priorPayload.CharPositionInLine;
							this.approximateLineInfo = true;
							break;
						}
						--i;
						priorNode = nodes.LT(i);
					}
				}
				else
				{
					// node created from real token
					this.line = payload.Line;
					this.charPositionInLine = payload.CharPositionInLine;
				}
			}
			else if (this.node is ITree)
			{
				this.line = ((ITree)this.node).Line;
				this.charPositionInLine = ((ITree)this.node).CharPositionInLine;
				if (this.node is CommonTree)
				{
					this.token = ((CommonTree)this.node).Token;
				}
			}
			else 
			{
				int type = adaptor.GetNodeType(this.node);
				string text = adaptor.GetNodeText(this.node);
				this.token = new CommonToken(type, text);
			}
		}

		#endregion

		#region Data Members


		/// <summary>What input stream did the error occur in? </summary>
		[NonSerialized]
		protected IIntStream input;

		/// <summary>
		/// What is index of token/char were we looking at when the error occurred?
		/// </summary>
		protected int index;

		/// <summary>
		/// The current Token when an error occurred.  Since not all streams
		/// can retrieve the ith Token, we have to track the Token object.
		/// </summary>
		protected IToken token;

		/// <summary>[Tree parser] Node with the problem.</summary>
		protected object node;

		/// <summary>The current char when an error occurred. For lexers. </summary>
		protected int c;

		/// <summary>Track the line at which the error occurred in case this is
		/// generated from a lexer.  We need to track this since the
		/// unexpected char doesn't carry the line info.
		/// </summary>
		protected int line;

		protected int charPositionInLine;

		/// <summary>
		/// If you are parsing a tree node stream, you will encounter some
		/// imaginary nodes w/o line/col info.  We now search backwards looking
		/// for most recent token with line/col info, but notify getErrorHeader()
		/// that info is approximate.
		/// </summary>
		public bool approximateLineInfo;

		#endregion
	}
}