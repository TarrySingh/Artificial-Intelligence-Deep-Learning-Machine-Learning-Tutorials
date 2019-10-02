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


namespace Antlr.Runtime.Debug
{
	using System;
	using Antlr.Runtime;
	using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;
	using ITreeNodeStream = Antlr.Runtime.Tree.ITreeNodeStream;


	/// <summary>
	/// Print out (most of) the events... Useful for debugging, testing...
	/// </summary>
	public class TraceDebugEventListener : BlankDebugEventListener
	{
		ITreeAdaptor adaptor;

		public TraceDebugEventListener(ITreeAdaptor adaptor)
		{
			this.adaptor = adaptor;
		}

		override public void EnterRule(string grammarFileName, string ruleName) {
			Console.Out.WriteLine("EnterRule " + grammarFileName + " " + ruleName);
		}
		override public void ExitRule(string grammarFileName, string ruleName) {
			Console.Out.WriteLine("ExitRule " + grammarFileName + " " + ruleName);
		}
		override public void EnterSubRule(int decisionNumber) { Console.Out.WriteLine("EnterSubRule"); }
		override public void ExitSubRule(int decisionNumber) { Console.Out.WriteLine("ExitSubRule"); }
		override public void Location(int line, int pos) { Console.Out.WriteLine("Location " + line + ":" + pos); }

		#region Tree parsing stuff

		override public void ConsumeNode(object t)
		{
			int ID = adaptor.GetUniqueID(t);
			string text = adaptor.GetNodeText(t);
			int type = adaptor.GetNodeType(t);
			Console.Out.WriteLine("ConsumeNode " + ID + " " + text + " " + type);
		}

		override public void LT(int i, object t)
		{
			int ID = adaptor.GetUniqueID(t);
			string text = adaptor.GetNodeText(t);
			int type = adaptor.GetNodeType(t);
			Console.Out.WriteLine("LT " + i + " " + ID + " " + text + " " + type);
		}

		#endregion

		#region AST stuff

		override public void GetNilNode(object t)
		{
			Console.Out.WriteLine("GetNilNode " + adaptor.GetUniqueID(t));
		}

		override public void CreateNode(object t)
		{
			int ID = adaptor.GetUniqueID(t);
			string text = adaptor.GetNodeText(t);
			int type = adaptor.GetNodeType(t);
			Console.Out.WriteLine("Create " + ID + ": " + text + ", " + type);
		}

		override public void CreateNode(object t, IToken token)
		{
			int ID = adaptor.GetUniqueID(t);
			//string text = adaptor.GetNodeText(t);
			int tokenIndex = token.TokenIndex;
			Console.Out.WriteLine("Create " + ID + ": " + tokenIndex);
		}

		override public void BecomeRoot(object newRoot, object oldRoot)
		{
			Console.Out.WriteLine("BecomeRoot " + adaptor.GetUniqueID(newRoot) + ", " + adaptor.GetUniqueID(oldRoot));
		}

		override public void AddChild(object root, object child)
		{
			Console.Out.WriteLine("AddChild " + adaptor.GetUniqueID(root) + ", " + adaptor.GetUniqueID(child));
		}

		override public void SetTokenBoundaries(object t, int tokenStartIndex, int tokenStopIndex)
		{
			Console.Out.WriteLine("SetTokenBoundaries " + adaptor.GetUniqueID(t) + ", " + tokenStartIndex + ", " + tokenStopIndex);
		}

		#endregion
	}
}

