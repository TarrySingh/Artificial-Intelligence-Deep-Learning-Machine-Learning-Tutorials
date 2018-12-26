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
	
	[Serializable]
	public class CommonToken : IToken
	{
		#region Constructors

		public CommonToken(int type)
		{
			this.type = type;
		}

		public CommonToken(ICharStream input, int type, int channel, int start, int stop)
		{
			this.input = input;
			this.type = type;
			this.channel = channel;
			this.start = start;
			this.stop = stop;
		}

		public CommonToken(int type, string text)
		{
			this.type = type;
			this.channel = Token.DEFAULT_CHANNEL;
			this.text = text;
		}

		public CommonToken(IToken oldToken)
		{
			text = oldToken.Text;
			type = oldToken.Type;
			line = oldToken.Line;
			index = oldToken.TokenIndex;
			charPositionInLine = oldToken.CharPositionInLine;
			channel = oldToken.Channel;
			if (oldToken is CommonToken) {
				start = ((CommonToken)oldToken).start;
				stop = ((CommonToken)oldToken).stop;
			}
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

		virtual public int StartIndex
		{
			get { return start; }
			set { this.start = value; }
		}

		virtual public int StopIndex
		{
			get { return stop; }
			set { this.stop = value; }
		}

		virtual public int TokenIndex
		{
			get { return index; }
			set { this.index = value; }
		}

		virtual public ICharStream InputStream
		{
			get { return input; }
			set { this.input = value; }
		}

		virtual public string Text
		{
			get
			{
				if (text != null)
				{
					return text;
				}
				if (input == null)
				{
					return null;
				}
				text = input.Substring(start, stop);
				return text;
			}
			set 
			{
				/* Override the text for this token.  The property getter 
				 * will return this text rather than pulling from the buffer.
				 * Note that this does not mean that start/stop indexes are 
				 * not valid.  It means that the input was converted to a new 
				 * string in the token object.
				 */
				this.text = value; 
			}
		}

		public override string ToString()
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
			return "[@" + TokenIndex + "," + start + ":" + stop + "='" + txt + "',<" + type + ">" + channelStr + "," + line + ":" + CharPositionInLine + "]";
		}

		#endregion

		#region Data Members

		protected internal int type;
		protected internal int line;
		protected internal int charPositionInLine = -1; // set to invalid position
		protected internal int channel = Token.DEFAULT_CHANNEL;
		[NonSerialized] protected internal ICharStream input;

		/// <summary>We need to be able to change the text once in a while.  If
		/// this is non-null, then getText should return this.  Note that
		/// start/stop are not affected by changing this.
		/// </summary>
		protected internal string text;

		/// <summary>What token number is this from 0..n-1 tokens; &lt; 0 implies invalid index </summary>
		protected internal int index = -1;

		/// <summary>The char position into the input buffer where this token starts </summary>
		protected internal int start;

		/// <summary>The char position into the input buffer where this token stops </summary>
		protected internal int stop;

		#endregion
	}
}
