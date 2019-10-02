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
	using System.Collections;
	using Stack					= System.Collections.Stack;
	using IToken				= Antlr.Runtime.IToken;
	using RecognitionException	= Antlr.Runtime.RecognitionException;
	using ParseTree				= Antlr.Runtime.Tree.ParseTree;


	/// <summary>
	/// This parser listener tracks rule entry/exit and token matches
	/// to build a simple parse tree using ParseTree nodes.
	/// </summary>
	public class ParseTreeBuilder : BlankDebugEventListener
	{
		public static readonly String EPSILON_PAYLOAD = "<epsilon>";

		Stack callStack = new Stack();
		IList hiddenTokens = new ArrayList();
		int backtracking = 0;

		public ParseTreeBuilder(string grammarName) 
		{
			ParseTree root = Create("<grammar " + grammarName + ">");
			callStack.Push(root);
		}

		public ParseTree Tree {
			get { return (ParseTree)callStack.Peek(); }
		}

		/// <summary>
		///  What kind of node to create.  You might want to override
		///  so I factored out creation here.
		/// </summary>
		public ParseTree Create(object payload) 
		{
			return new ParseTree(payload);
		}

		public ParseTree EpsilonNode() {
			return Create(EPSILON_PAYLOAD);
		}

		/** Backtracking or cyclic DFA, don't want to add nodes to tree */
		override public void EnterDecision(int d) { backtracking++; }
		override public void ExitDecision(int i) { backtracking--; }

		override public void EnterRule(string filename, string ruleName) {
			if ( backtracking>0 ) return;
			ParseTree parentRuleNode = (ParseTree)callStack.Peek();
			ParseTree ruleNode = Create(ruleName);
			parentRuleNode.AddChild(ruleNode);
			callStack.Push(ruleNode);
		}

		override public void ExitRule(string filename, string ruleName) 
		{
			if ( backtracking>0 ) return;
			ParseTree ruleNode = (ParseTree)callStack.Peek();
			if ( ruleNode.ChildCount==0 ) {
				ruleNode.AddChild(EpsilonNode());
			}
			callStack.Pop();
		}

		override public void ConsumeToken(IToken token) 
		{
			if ( backtracking>0 ) return;
			ParseTree ruleNode = (ParseTree)callStack.Peek();
			ParseTree elementNode = Create(token);
			elementNode.hiddenTokens = this.hiddenTokens;
			this.hiddenTokens = new ArrayList();
			ruleNode.AddChild(elementNode);
		}

		override public void ConsumeHiddenToken(IToken token) {
			if ( backtracking>0 ) return;
			hiddenTokens.Add(token);
		}

		override public void RecognitionException(RecognitionException e) 
		{
			if ( backtracking>0 ) return;
			ParseTree ruleNode = (ParseTree)callStack.Peek();
			ParseTree errorNode = Create(e);
			ruleNode.AddChild(errorNode);
		}
	}
}