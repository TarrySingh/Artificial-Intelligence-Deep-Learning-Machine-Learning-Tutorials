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
	using IToken = Antlr.Runtime.IToken;
	using RecognitionException = Antlr.Runtime.RecognitionException;

	/// <summary>
	/// A simple event repeater (proxy) that delegates all functionality to 
	/// the listener sent into the ctor.
	/// </summary>
	/// <remarks>
	/// Useful if you want to listen in on a few debug events w/o 
	/// interrupting the debugger.  Just subclass the repeater and override 
	/// the methods you want to listen in on.  Remember to call the method 
	/// in this class so the event will continue on to the original recipient.
	/// </remarks>
	public class DebugEventRepeater : IDebugEventListener
	{
		protected IDebugEventListener listener;

		public DebugEventRepeater(IDebugEventListener listener)
		{
			this.listener = listener;
		}

		public void EnterRule(string grammarFileName, string ruleName) { listener.EnterRule(grammarFileName, ruleName); }
		public void ExitRule(string grammarFileName, string ruleName) { listener.ExitRule(grammarFileName, ruleName); }
		public void EnterAlt(int alt) { listener.EnterAlt(alt); }
		public void EnterSubRule(int decisionNumber) { listener.EnterSubRule(decisionNumber); }
		public void ExitSubRule(int decisionNumber) { listener.ExitSubRule(decisionNumber); }
		public void EnterDecision(int decisionNumber) { listener.EnterDecision(decisionNumber); }
		public void ExitDecision(int decisionNumber) { listener.ExitDecision(decisionNumber); }
		public void Location(int line, int pos) { listener.Location(line, pos); }
		public void ConsumeToken(IToken token) { listener.ConsumeToken(token); }
		public void ConsumeHiddenToken(IToken token) { listener.ConsumeHiddenToken(token); }
		public void LT(int i, IToken t) { listener.LT(i, t); }
		public void Mark(int i) { listener.Mark(i); }
		public void Rewind(int i) { listener.Rewind(i); }
		public void Rewind() { listener.Rewind(); }
		public void BeginBacktrack(int level) { listener.BeginBacktrack(level); }
		public void EndBacktrack(int level, bool successful) { listener.EndBacktrack(level, successful); }
		public void RecognitionException(RecognitionException e) { listener.RecognitionException(e); }
		public void BeginResync() { listener.BeginResync(); }
		public void EndResync() { listener.EndResync(); }
		public void SemanticPredicate(bool result, string predicate) { listener.SemanticPredicate(result, predicate); }
		public void Commence() { listener.Commence(); }
		public void Terminate() { listener.Terminate(); }

		// Tree parsing stuff

		public void ConsumeNode(object t) { listener.ConsumeNode(t); }
		public void LT(int i, object t) { listener.LT(i, t); }

		// AST Stuff

		public void GetNilNode(object t) { listener.GetNilNode(t); }
		public void ErrorNode(object t) { listener.ErrorNode(t); }
		public void CreateNode(object t) { listener.CreateNode(t); }
		public void CreateNode(object node, IToken token) { listener.CreateNode(node, token); }
		public void BecomeRoot(object newRoot, object oldRoot) { listener.BecomeRoot(newRoot, oldRoot); }
		public void AddChild(object root, object child) { listener.AddChild(root, child); }
		public void SetTokenBoundaries(object t, int tokenStartIndex, int tokenStopIndex)
		{
			listener.SetTokenBoundaries(t, tokenStartIndex, tokenStopIndex);
		}
	}
}