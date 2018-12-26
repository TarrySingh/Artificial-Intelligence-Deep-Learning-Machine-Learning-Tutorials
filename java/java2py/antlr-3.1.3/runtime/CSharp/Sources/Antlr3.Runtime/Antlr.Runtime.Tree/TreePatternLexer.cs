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


namespace Antlr.Runtime.Tree
{
	using System;
	using StringBuilder = System.Text.StringBuilder;

	public class TreePatternLexer
	{
		public const int EOF = -1;
		public const int BEGIN = 1;
		public const int END = 2;
		public const int ID = 3;
		public const int ARG = 4;
		public const int PERCENT = 5;
		public const int COLON = 6;
		public const int DOT = 7;

		/// <summary>The tree pattern to lex like "(A B C)"</summary>
		protected string pattern;

		/// <summary>Index into input string</summary>
		protected int p = -1;

		/// <summary>Current char</summary>
		protected int c;

		/// <summary>How long is the pattern in char?</summary>
		protected int n;

		/// <summary>
		/// Set when token type is ID or ARG (name mimics Java's StreamTokenizer)
		/// </summary>
		public StringBuilder sval = new StringBuilder();

		public bool error = false;

		public TreePatternLexer(string pattern)
		{
			this.pattern = pattern;
			this.n = pattern.Length;
			Consume();
		}

		public int NextToken()
		{
			sval.Length = 0; // reset, but reuse buffer
			while (c != EOF)
			{
				if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
				{
					Consume();
					continue;
				}
				if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_')
				{
					sval.Append((char)c);
					Consume();
					while ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
							(c >= '0' && c <= '9') || c == '_')
					{
						sval.Append((char)c);
						Consume();
					}
					return ID;
				}
				if (c == '(')
				{
					Consume();
					return BEGIN;
				}
				if (c == ')')
				{
					Consume();
					return END;
				}
				if (c == '%')
				{
					Consume();
					return PERCENT;
				}
				if (c == ':')
				{
					Consume();
					return COLON;
				}
				if (c == '.')
				{
					Consume();
					return DOT;
				}
				if (c == '[')
				{ // grab [x] as a string, returning x
					Consume();
					while (c != ']')
					{
						if (c == '\\')
						{
							Consume();
							if (c != ']')
							{
								sval.Append('\\');
							}
							sval.Append((char)c);
						}
						else
						{
							sval.Append((char)c);
						}
						Consume();
					}
					Consume();
					return ARG;
				}
				Consume();
				error = true;
				return EOF;
			}
			return EOF;
		}

		protected void Consume()
		{
			p++;
			if (p >= n)
			{
				c = EOF;
			}
			else
			{
				c = pattern[p];
			}
		}
	}
}