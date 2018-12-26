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
	using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;

	/// <summary>
	/// A TreeAdaptor proxy that fires debugging events to a DebugEventListener 
	/// delegate and uses the TreeAdaptor delegate to do the actual work.  All
	/// AST events are triggered by this adaptor; no code gen changes are needed
	/// in generated rules.  Debugging events are triggered *after* invoking
	/// tree adaptor routines.
	///
	/// Trees created with actions in rewrite actions like "-> ^(ADD {foo} {bar})"
	/// cannot be tracked as they might not use the adaptor to create foo, bar.
	/// The debug listener has to deal with tree node IDs for which it did
	/// not see a CreateNode event.  A single &lt;unknown&gt; node is sufficient even
	/// if it represents a whole tree.
	/// </summary>
	public class DebugTreeAdaptor : ITreeAdaptor
	{
		protected IDebugEventListener dbg;
		protected ITreeAdaptor adaptor;

		public DebugTreeAdaptor(IDebugEventListener dbg, ITreeAdaptor adaptor)
		{
			this.dbg = dbg;
			this.adaptor = adaptor;
		}

		public object Create(IToken payload)
		{
			if (payload.TokenIndex < 0) {
				// could be token conjured up during error recovery
				return Create(payload.Type, payload.Text);
			}
			object node = adaptor.Create(payload);
			dbg.CreateNode(node, payload);
			return node;
		}

		public Object ErrorNode(ITokenStream input, IToken start, IToken stop,
								RecognitionException e)
		{
			Object node = adaptor.ErrorNode(input, start, stop, e);
			if (node != null) {
				dbg.ErrorNode(node);
			}
			return node;
		}

		public object DupTree(object tree)
		{
			Object t = adaptor.DupTree(tree);
			// walk the tree and emit create and add child events
			// to simulate what DupTree has done. DupTree does not call this debug
			// adapter so I must simulate.
			SimulateTreeConstruction(t);
			return t;
		}
	
		/** ^(A B C): emit create A, create B, add child, ...*/
		protected void SimulateTreeConstruction(Object t) {
			dbg.CreateNode(t);
			int n = adaptor.GetChildCount(t);
			for (int i=0; i<n; i++) {
				Object child = adaptor.GetChild(t, i);
				SimulateTreeConstruction(child);
				dbg.AddChild(t, child);
			}
		}

		public object DupNode(object treeNode)
		{
			Object d = adaptor.DupNode(treeNode);
			dbg.CreateNode(d);
			return d;
		}

		public object GetNilNode()
		{
			object node = adaptor.GetNilNode();
			dbg.GetNilNode(node);
			return node;
		}

		public bool IsNil(object tree)
		{
			return adaptor.IsNil(tree);
		}

		public void AddChild(object t, object child)
		{
			if ((t == null) || (child == null))
			{
				return;
			}
			adaptor.AddChild(t, child);
			dbg.AddChild(t, child);
		}

		public object BecomeRoot(object newRoot, object oldRoot)
		{
			object n = adaptor.BecomeRoot(newRoot, oldRoot);
			dbg.BecomeRoot(newRoot, oldRoot);
			return n;
		}

		public object RulePostProcessing(object root)
		{
			return adaptor.RulePostProcessing(root);
		}

		public void AddChild(object t, IToken child)
		{
			object n = this.Create(child);
			this.AddChild(t, n);
		}

		public object BecomeRoot(IToken newRoot, object oldRoot)
		{
			object n = this.Create(newRoot);
			adaptor.BecomeRoot(n, oldRoot);
			dbg.BecomeRoot(newRoot, oldRoot);
			return n;
		}

		public object Create(int tokenType, IToken fromToken)
		{
			object node = adaptor.Create(tokenType, fromToken);
			dbg.CreateNode(node);
			return node;
		}

		public object Create(int tokenType, IToken fromToken, string text)
		{
			object node = adaptor.Create(tokenType, fromToken, text);
			dbg.CreateNode(node);
			return node;
		}

		public object Create(int tokenType, string text)
		{
			object node = adaptor.Create(tokenType, text);
			dbg.CreateNode(node);
			return node;
		}

		public int GetNodeType(object t)
		{
			return adaptor.GetNodeType(t);
		}

		public void SetNodeType(object t, int type)
		{
			adaptor.SetNodeType(t, type);
		}

		public string GetNodeText(object t)
		{
			return adaptor.GetNodeText(t);
		}

		public void SetNodeText(object t, string text)
		{
			adaptor.SetNodeText(t, text);
		}

		public IToken GetToken(object treeNode)
		{
			return adaptor.GetToken(treeNode);
		}

		public void SetTokenBoundaries(object t, IToken startToken, IToken stopToken)
		{
			adaptor.SetTokenBoundaries(t, startToken, stopToken);
			if ( (t != null) && (startToken != null) && (stopToken != null) )
			{
				dbg.SetTokenBoundaries(t,
									   startToken.TokenIndex,
									   stopToken.TokenIndex);
			}
		}

		public int GetTokenStartIndex(object t)
		{
			return adaptor.GetTokenStartIndex(t);
		}

		public int GetTokenStopIndex(object t)
		{
			return adaptor.GetTokenStopIndex(t);
		}

		public object GetChild(object t, int i)
		{
			return adaptor.GetChild(t, i);
		}

		public void SetChild(object t, int i, object child)
		{
			adaptor.SetChild(t, i, child);
		}

		public object DeleteChild(object t, int i)
		{
			return adaptor.DeleteChild(t, i);
		}

		public int GetChildCount(object t)
		{
			return adaptor.GetChildCount(t);
		}

		public int GetUniqueID(object node)
		{
			return adaptor.GetUniqueID(node);
		}

		public object GetParent(object t)
		{
			return adaptor.GetParent(t);
		}

		public int GetChildIndex(object t)
		{
			return adaptor.GetChildIndex(t);
		}

		public void SetParent(object t, object parent)
		{
			adaptor.SetParent(t, parent);
		}

		public void SetChildIndex(object t, int index)
		{
			adaptor.SetChildIndex(t, index);
		}

		public void ReplaceChildren(object parent, int startChildIndex, int stopChildIndex, object t)
		{
			adaptor.ReplaceChildren(parent, startChildIndex, stopChildIndex, t);
		}

		#region Support

		public IDebugEventListener DebugListener
		{
			get { return dbg;  }
			set { dbg = value; }
		}

		public ITreeAdaptor TreeAdaptor
		{
			get { return adaptor; }
		}

		#endregion
	}
}