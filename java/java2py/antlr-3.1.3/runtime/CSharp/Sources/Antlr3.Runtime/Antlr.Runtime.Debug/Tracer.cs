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
	/// The default tracer mimics the traceParser behavior of ANTLR 2.x.
	/// This listens for debugging events from the parser and implies
	/// that you cannot debug and trace at the same time.
	/// </summary>
	public class Tracer : BlankDebugEventListener
	{
		public IIntStream input;
		protected int level = 0;
		
		public Tracer(IIntStream input)
		{
			this.input = input;
		}
		
		override public void EnterRule(string grammarFileName, string ruleName)
		{
			for (int i = 1; i <= level; i++)
			{
				Console.Out.Write(" ");
			}
			Console.Out.WriteLine("> " + grammarFileName + " " + ruleName + " lookahead(1)=" + GetInputSymbol(1));
			level++;
		}

		override public void ExitRule(string grammarFileName, string ruleName)
		{
			level--;
			for (int i = 1; i <= level; i++)
			{
				Console.Out.Write(" ");
			}
			Console.Out.WriteLine("< " + grammarFileName + " " + ruleName + " lookahead(1)=" + GetInputSymbol(1));
		}

		public virtual object GetInputSymbol(int k)
		{
			if (input is ITokenStream)
			{
				return ((ITokenStream) input).LT(k);
			}
			return (char) input.LA(k);
		}
	}
}