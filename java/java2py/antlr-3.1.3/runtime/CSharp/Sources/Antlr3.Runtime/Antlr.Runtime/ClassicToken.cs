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
	
	/// <summary>
	/// A Token object like we'd use in ANTLR 2.x; has an actual string created
	/// and associated with this object.  These objects are needed for imaginary
	/// tree nodes that have payload objects.  We need to create a Token object
	/// that has a string; the tree node will point at this token.  CommonToken
	/// has indexes into a char stream and hence cannot be used to introduce
	/// new strings.
	/// </summary>
	[Serializable]
	public class ClassicToken : IToken
	{
		#region Constructors

		public ClassicToken(int type)
		{
			this.type = type;
		}

		public ClassicToken(IToken oldToken)
		{
			text = oldToken.Text;
			type = oldToken.Type;
			line = oldToken.Line;
			charPositionInLine = oldToken.CharPositionInLine;
			channel = oldToken.Channel;
		}

		public ClassicToken(int type, string text)
		{
			this.type = type;
			this.text = text;
		}

		public ClassicToken(int type, string text, int channel)
		{
			this.type = type;
			this.text = text;
			this.channel = channel;
		}

		#endregion

		#region Public API

		virtual public int Type
		{
			get { return type; }
			set { this.type = value; }
		}

		virtual public int Line
		{
			get { return line; }
			set { this.line = value; }
		}

		virtual public int CharPositionInLine
		{
			get { return charPositionInLine; }
			set { this.charPositionInLine = value; }
		}

		virtual public int Channel
		{
			get { return channel; }
			set { this.channel = value; }
		}

		virtual public int TokenIndex
		{
			get { return index; }
			set { this.index = value; }
		}

		virtual public string Text
		{
			get { return text; }
			set { text = value; }
		}

		virtual public ICharStream InputStream
		{
			get { return null; }
			set { }
		}

		override public string ToString()
		{
			string channelStr = "";
			if (channel > 0)
			{
				channelStr = ",channel=" + channel;
			}
			string txt = Text;
			if (txt != null)
			{
				txt = txt.Replace("\n", "\\\\n");
				txt = txt.Replace("\r", "\\\\r");
				txt = txt.Replace("\t", "\\\\t");
			}
			else
			{
				txt = "<no text>";
			}
			return "[@" + TokenIndex + ",'" + txt + "',<" + type + ">" + channelStr + "," + line + ":" + CharPositionInLine + "]";
		}

		#endregion

		#region Data Members

		protected internal string text;
		protected internal int type;
		protected internal int line;
		protected internal int charPositionInLine;
		protected internal int channel = Token.DEFAULT_CHANNEL;

		/// <summary>What token number is this from 0..n-1 tokens </summary>
		protected internal int index;

		#endregion
	}
}
