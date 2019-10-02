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
	using ITreeAdaptor		= Antlr.Runtime.Tree.ITreeAdaptor;
	using ITreeNodeStream	= Antlr.Runtime.Tree.ITreeNodeStream;
	using ITokenStream		= Antlr.Runtime.ITokenStream;


	/// <summary>
	/// Debug any tree node stream.  The constructor accepts the stream
	/// and a debug listener.  As node stream calls come in, debug events
	/// are triggered.
	/// </summary>
	public class DebugTreeNodeStream : ITreeNodeStream
	{
		protected IDebugEventListener dbg;
		protected ITreeAdaptor adaptor;
		protected ITreeNodeStream input;
		protected bool initialStreamState = true;

		/// <summary>Track the last mark() call result value for use in rewind().</summary>
		protected int lastMarker;

		public DebugTreeNodeStream(ITreeNodeStream input, IDebugEventListener dbg)
		{
			this.input = input;
			this.adaptor = input.TreeAdaptor;
			this.input.HasUniqueNavigationNodes = true;
			SetDebugListener(dbg);
		}

		public void SetDebugListener(IDebugEventListener dbg)
		{
			this.dbg = dbg;
		}

		public ITokenStream TokenStream
		{
			get { return input.TokenStream; }
		}

		public string SourceName {
			get { return TokenStream.SourceName; }
		}

		public ITreeAdaptor TreeAdaptor
		{
			get { return adaptor; }
		}

		public void Consume()
		{
			object node = input.LT(1);
			input.Consume();
			dbg.ConsumeNode(node);
		}

		public object Get(int i) 
		{
			return input.Get(i);
		}

		public object LT(int i)
		{
			object node = input.LT(i);
			dbg.LT(i, node);
			return node;
		}

		public int LA(int i)
		{
			object node = input.LT(i);
			int type = adaptor.GetNodeType(node);
			dbg.LT(i, node);
			return type;
		}

		public int Mark()
		{
			lastMarker = input.Mark();
			dbg.Mark(lastMarker);
			return lastMarker;
		}

		public int Index()
		{
			return input.Index();
		}

		public void Rewind(int marker)
		{
			dbg.Rewind(marker);
			input.Rewind(marker);
		}

		public void Rewind()
		{
			dbg.Rewind();
			input.Rewind(lastMarker);
		}

		public void Release(int marker)
		{
		}

		public void Seek(int index)
		{
			input.Seek(index);
		}

		[Obsolete("Please use property Count instead.")]
		public int Size()
		{
			return Count;
		}
		
		public int Count {
			get { return input.Count; }
		}

		public object TreeSource
		{
			get { return input; }
		}

		/// <summary>
		/// It is normally this object that instructs the node stream to
		/// create unique nav nodes, but to satisfy interface, we have to
		/// define it.  It might be better to ignore the parameter but
		/// there might be a use for it later, so I'll leave.
		/// </summary>
		public virtual bool HasUniqueNavigationNodes
		{
			set { input.HasUniqueNavigationNodes = value; }
		}

		public void ReplaceChildren(object parent, int startChildIndex, int stopChildIndex, object t)
		{
			input.ReplaceChildren(parent, startChildIndex, stopChildIndex, t);
		}

		public string ToString(object start, object stop)
		{
			return input.ToString(start, stop);
		}
	}
}