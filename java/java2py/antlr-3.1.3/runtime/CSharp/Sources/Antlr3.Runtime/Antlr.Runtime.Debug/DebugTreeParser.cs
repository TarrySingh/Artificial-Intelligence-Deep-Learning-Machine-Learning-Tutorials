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
	using System.IO;
	using Antlr.Runtime;
	using TreeParser = Antlr.Runtime.Tree.TreeParser;
	using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;
	using ITreeNodeStream = Antlr.Runtime.Tree.ITreeNodeStream;
	using ErrorManager = Antlr.Runtime.Misc.ErrorManager;

	public class DebugTreeParser : TreeParser
	{
		/// <summary>Who to notify when events in the parser occur.</summary>
		protected IDebugEventListener dbg = null;

		/// <summary>
		/// Used to differentiate between fixed lookahead and cyclic DFA decisions
		/// while profiling.
		/// </summary>
		public bool isCyclicDecision = false;

		/// <summary>
		/// Create a normal parser except wrap the token stream in a debug
		/// proxy that fires consume events.
		/// </summary>
		public DebugTreeParser(ITreeNodeStream input, IDebugEventListener dbg, RecognizerSharedState state)
			: base((input is DebugTreeNodeStream ? input : new DebugTreeNodeStream(input, dbg)), state)
		{
			
			DebugListener = dbg;
		}

		public DebugTreeParser(ITreeNodeStream input, RecognizerSharedState state)
			: base((input is DebugTreeNodeStream ? input : new DebugTreeNodeStream(input, null)), state)
		{
		}

		public DebugTreeParser(ITreeNodeStream input, IDebugEventListener dbg)
			: this((input is DebugTreeNodeStream ? input : new DebugTreeNodeStream(input, dbg)), dbg, null)
		{
		}

		/// <summary>
		/// Provide a new debug event listener for this parser.  Notify the
		/// input stream too that it should send events to this listener.
		/// </summary>
		public IDebugEventListener DebugListener
		{
			get { return dbg; }
			set
			{
				if (input is DebugTreeNodeStream)
				{
					((DebugTreeNodeStream)input).SetDebugListener(value);
				}
				this.dbg = value;
			}
		}

		public virtual void ReportError(IOException e)
		{
			ErrorManager.InternalError(e);
		}

		public override void ReportError(RecognitionException e) {
			dbg.RecognitionException(e);
		}

		protected override object GetMissingSymbol(IIntStream input,
										  RecognitionException e,
										  int expectedTokenType,
										  BitSet follow) {
			object o = base.GetMissingSymbol(input, e, expectedTokenType, follow);
			dbg.ConsumeNode(o);
			return o;
		}

		override public void BeginResync()
		{
			dbg.BeginResync();
		}

		override public void EndResync()
		{
			dbg.EndResync();
		}

		override public void BeginBacktrack(int level)
		{
			dbg.BeginBacktrack(level);
		}

		override public void EndBacktrack(int level, bool successful)
		{
			dbg.EndBacktrack(level, successful);
		}
	}
}