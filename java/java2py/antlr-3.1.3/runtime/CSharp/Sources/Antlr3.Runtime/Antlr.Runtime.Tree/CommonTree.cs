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


namespace Antlr.Runtime.Tree
{
	using System;
	using IToken = Antlr.Runtime.IToken;
	
	/// <summary>A tree node that is wrapper for a Token object. </summary>
	/// <remarks>
	/// After 3.0 release while building tree rewrite stuff, it became clear 
	/// that computing parent and child index is very difficult and cumbersome.
	/// Better to spend the space in every tree node.  If you don't want these 
	/// extra fields, it's easy to cut them out in your own BaseTree subclass.
	/// </remarks>
	[Serializable]
	public class CommonTree : BaseTree
	{
		public CommonTree()
		{
		}
		
		public CommonTree(CommonTree node) 
			: base(node)
		{
			this.token = node.token;
			this.startIndex = node.startIndex;
			this.stopIndex = node.stopIndex;
		}
		
		public CommonTree(IToken t)
		{
			this.token = t;
		}
		
		virtual public IToken Token
		{
			get { return token; }
		}

		override public bool IsNil
		{
			get { return token == null; }
		}

		override public int Type
		{
			get
			{
				if (token == null)
				{
					return Runtime.Token.INVALID_TOKEN_TYPE;
				}
				return token.Type;
			}
		}

		override public string Text
		{
			get 
			{
				if (token == null)
				{
					return null;
				}
				return token.Text;
			}
		}

		override public int Line
		{
			get
			{
				if (token == null || token.Line == 0)
				{
					if (ChildCount > 0)
					{
						return GetChild(0).Line;
					}
					return 0;
				}
				return token.Line;
			}
		}

		override public int CharPositionInLine
		{
			get
			{
				if (token == null || token.CharPositionInLine == - 1)
				{
					if (ChildCount > 0)
					{
						return GetChild(0).CharPositionInLine;
					}
					return 0;
				}
				return token.CharPositionInLine;
			}
		}

		override public int TokenStartIndex
		{
			get
			{
				if ( (startIndex == -1) && (token != null) ) 
				{
					return token.TokenIndex;
				}
				return startIndex;
			}

			set { startIndex = value; }
		}

		override public int TokenStopIndex
		{
			get
			{
				if ( (stopIndex == -1) && (token != null) ) 
				{
					return token.TokenIndex;
				}
				return stopIndex;
			}

			set { stopIndex = value; }
		}

		/// <summary>
		/// For every node in this subtree, make sure it's start/stop token's
	    /// are set.  Walk depth first, visit bottom up.  Only updates nodes
	    /// with at least one token index < 0.
		/// </summary>
	    public void SetUnknownTokenBoundaries() {
	        if ( children==null ) {
	            if ( startIndex<0 || stopIndex<0 ) {
	                startIndex = stopIndex = token.TokenIndex;
	            }
	            return;
	        }
	        for (int i=0; i<children.Count; i++) {
	            ((CommonTree)children[i]).SetUnknownTokenBoundaries();
	        }
	        if ( startIndex>=0 && stopIndex>=0 ) return; // already set
	        if ( children.Count > 0 ) {
	            CommonTree firstChild = (CommonTree)children[0];
	            CommonTree lastChild = (CommonTree)children[children.Count-1];
	            startIndex = firstChild.TokenStartIndex;
	            stopIndex = lastChild.TokenStopIndex;
	        }
	    }

		override public int ChildIndex
		{
			get { return childIndex;  }
			set { childIndex = value; }
		}

		override public ITree Parent
		{
			get { return parent; }
			set { parent = (CommonTree)value; }
		}

		/// <summary>
		/// What token indexes bracket all tokens associated with this node
		/// and below?
		/// </summary>
		public int startIndex = -1, stopIndex = -1;
		
		/// <summary>A single token is the payload </summary>
		protected IToken token;

		/// <summary>Who is the parent node of this node; if null, implies node is root</summary>
		public CommonTree parent;

		/// <summary>What index is this node in the child list? Range: 0..n-1</summary>
		public int childIndex = -1;

		public override ITree DupNode()
		{
			return new CommonTree(this);
		}
		
		public override string ToString()
		{
			if (IsNil)
			{
				return "nil";
			}
			if ( Type == Runtime.Token.INVALID_TOKEN_TYPE ) {
				return "<errornode>";
			}
			if (token == null) {
				return null;
			}
			return token.Text;
		}
	}
}