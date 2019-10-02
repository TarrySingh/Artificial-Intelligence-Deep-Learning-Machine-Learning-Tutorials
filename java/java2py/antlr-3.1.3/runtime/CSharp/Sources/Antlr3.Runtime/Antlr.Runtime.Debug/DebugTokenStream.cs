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


namespace Antlr.Runtime.Debug
{
	using System;
	using Antlr.Runtime;
	
	public class DebugTokenStream : ITokenStream
	{
		protected internal IDebugEventListener dbg;
		public ITokenStream input;
		protected internal bool initialStreamState = true;
		/// <summary>
		/// Track the last Mark() call result value for use in Rewind().
		/// </summary>
		protected int lastMarker;

		virtual public IDebugEventListener DebugListener
		{
			set { this.dbg = value; }
		}
		
		public DebugTokenStream(ITokenStream input, IDebugEventListener dbg)
		{
			this.input = input;
			DebugListener = dbg;
			// force TokenStream to get at least first valid token
			// so we know if there are any hidden tokens first in the stream
			input.LT(1);
		}
		
		public virtual void  Consume()
		{
			if (initialStreamState)
			{
				ConsumeInitialHiddenTokens();
			}
			int a = input.Index();
			IToken t = input.LT(1);
			input.Consume();
			int b = input.Index();
			dbg.ConsumeToken(t);
			if (b > a + 1)
			{
				// then we consumed more than one token; must be off channel tokens
				for (int i = a + 1; i < b; i++)
				{
					dbg.ConsumeHiddenToken(input.Get(i));
				}
			}
		}
		
		/// <summary>consume all initial off-channel tokens</summary>
		protected internal virtual void ConsumeInitialHiddenTokens()
		{
			int firstOnChannelTokenIndex = input.Index();
			for (int i = 0; i < firstOnChannelTokenIndex; i++)
			{
				dbg.ConsumeHiddenToken(input.Get(i));
			}
			initialStreamState = false;
		}
		
		public virtual IToken LT(int i)
		{
			if (initialStreamState)
			{
				ConsumeInitialHiddenTokens();
			}
			dbg.LT(i, input.LT(i));
			return input.LT(i);
		}
		
		public virtual int LA(int i)
		{
			if (initialStreamState)
			{
				ConsumeInitialHiddenTokens();
			}
			dbg.LT(i, input.LT(i));
			return input.LA(i);
		}
		
		public virtual IToken Get(int i)
		{
			return input.Get(i);
		}
		
		public virtual int Mark()
		{
			lastMarker = input.Mark();
			dbg.Mark(lastMarker);
			return lastMarker;
		}
		
		public virtual int Index()
		{
			return input.Index();
		}
		
		public virtual void  Rewind(int marker)
		{
			dbg.Rewind(marker);
			input.Rewind(marker);
		}

		public virtual void Rewind()
		{
			dbg.Rewind();
			input.Rewind(lastMarker);
		}

		public virtual void Release(int marker)
		{
		}
		
		public virtual void  Seek(int index)
		{
			input.Seek(index);
		}
		
		[Obsolete("Please use property Count instead.")]
		public virtual int Size()
		{
			return Count;
		}
		
		public virtual int Count {
			get { return input.Count; }
		}

		public virtual ITokenSource TokenSource
		{
			get { return input.TokenSource; }
		}
		
		public virtual string SourceName {
			get { return TokenSource.SourceName; }
		}

		public override string ToString()
		{
			return input.ToString();
		}
		
		public virtual string ToString(int start, int stop)
		{
			return input.ToString(start, stop);
		}
		
		public virtual string ToString(IToken start, IToken stop)
		{
			return input.ToString(start, stop);
		}
	}
}