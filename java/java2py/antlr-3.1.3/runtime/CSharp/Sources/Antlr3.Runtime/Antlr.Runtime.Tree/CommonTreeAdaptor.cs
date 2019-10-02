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


namespace Antlr.Runtime.Tree
{
	using System;
	using CommonToken = Antlr.Runtime.CommonToken;
	using IToken = Antlr.Runtime.IToken;
	
	/// <summary>
	/// A TreeAdaptor that works with any Tree implementation.  It provides
	/// really just factory methods; all the work is done by BaseTreeAdaptor.
	/// If you would like to have different tokens created than ClassicToken
	/// objects, you need to override this and then set the parser tree adaptor to
	/// use your subclass.
	/// 
	/// To get your parser to build nodes of a different type, override
	/// Create(Token), ErrorNode(), and to be safe, YourTreeClass.DupNode().
 	/// DupNode() is called to duplicate nodes during rewrite operations.
	/// </summary>
	public class CommonTreeAdaptor : BaseTreeAdaptor
	{
		/// <summary>
		/// Duplicate a node.  This is part of the factory;
		/// override if you want another kind of node to be built.
		/// 
		/// I could use reflection to prevent having to override this
		/// but reflection is slow.
		/// </summary>
		public override object DupNode(object t)
		{
			if (t == null)
			{
				return null;
			}
			return ((ITree)t).DupNode();
		}
		
		public override object Create(IToken payload)
		{
			return new CommonTree(payload);
		}
		
		/// <summary>Create an imaginary token from a type and text </summary>
		/// <remarks>
		/// Tell me how to create a token for use with imaginary token nodes.
		/// For example, there is probably no input symbol associated with imaginary
		/// token DECL, but you need to create it as a payload or whatever for
		/// the DECL node as in ^(DECL type ID).
		/// 
		/// If you care what the token payload objects' type is, you should
		/// override this method and any other createToken variant.
		/// </remarks>
		public override IToken CreateToken(int tokenType, string text)
		{
			return new CommonToken(tokenType, text);
		}
		
		/// <summary>Create an imaginary token, copying the contents of a previous token </summary>
		/// <remarks>
		/// Tell me how to create a token for use with imaginary token nodes.
		/// For example, there is probably no input symbol associated with imaginary
		/// token DECL, but you need to create it as a payload or whatever for
		/// the DECL node as in ^(DECL type ID).
		/// 
		/// This is a variant of createToken where the new token is derived from
		/// an actual real input token.  Typically this is for converting '{'
		/// tokens to BLOCK etc...  You'll see
		/// 
		///    r : lc='{' ID+ '}' -> ^(BLOCK[$lc] ID+) ;
		/// 
		/// If you care what the token payload objects' type is, you should
		/// override this method and any other createToken variant.
		/// </remarks>
		public override IToken CreateToken(IToken fromToken)
		{
			return new CommonToken(fromToken);
		}
		
		/// <summary>track start/stop token for subtree root created for a rule </summary>
		/// <remarks>
		/// Track start/stop token for subtree root created for a rule.
		/// Only works with Tree nodes.  For rules that match nothing,
		/// seems like this will yield start=i and stop=i-1 in a nil node.
		/// Might be useful info so I'll not force to be i..i.
		/// </remarks>
		public override void  SetTokenBoundaries(object t, IToken startToken, IToken stopToken)
		{
			if (t == null)
			{
				return ;
			}

			int start = 0;
			int stop = 0;
			if (startToken != null)
			{
				start = startToken.TokenIndex;
			}
			if (stopToken != null)
			{
				stop = stopToken.TokenIndex;
			}
			((ITree) t).TokenStartIndex = start;
			((ITree) t).TokenStopIndex  = stop;
		}

		override public int GetTokenStartIndex(object t)
		{
			if (t == null)
			{
				return -1;
			}
			return ((ITree)t).TokenStartIndex;
		}

		override public int GetTokenStopIndex(object t)
		{
			if (t == null)
			{
				return -1;
			}
			return ((ITree)t).TokenStopIndex;
		}

		override public string GetNodeText(object t)
		{
			if (t == null)
			{
				return null;
			}
			return ((ITree)t).Text;
		}

		override public int GetNodeType(object t)
		{
			if (t == null)
			{
				return Token.INVALID_TOKEN_TYPE;
			}
			return ((ITree)t).Type;
		}

		/// <summary>
		/// What is the Token associated with this node?
		/// </summary>
		/// <remarks>
		/// If you are not using CommonTree, then you must override this in your 
		/// own adaptor.
		/// </remarks>
		override public IToken GetToken(object treeNode)
		{
			if ( treeNode is CommonTree ) 
			{
				return ((CommonTree)treeNode).Token;
			}
			return null; // no idea what to do
		}

		override public object GetChild(object t, int i) 
		{
			if (t == null)
			{
				return null;
			}
			return ((ITree)t).GetChild(i);
		}

		override public int GetChildCount(object t) 
		{
			if (t == null)
			{
				return 0;
			}
			return ((ITree)t).ChildCount;
		}

		override public object GetParent(object t)
		{
			if ( t==null ) return null;
			return ((ITree)t).Parent;
		}

		override public void SetParent(object t, object parent)
		{
			if ( t==null ) ((ITree)t).Parent = (ITree)parent;
		}

		override public int GetChildIndex(object t)
		{
			if ( t==null ) return 0;
			return ((ITree)t).ChildIndex;
		}

		override public void SetChildIndex(object t, int index)
		{
			if ( t==null ) ((ITree)t).ChildIndex = index;
		}

		override public void ReplaceChildren(object parent, int startChildIndex, int stopChildIndex, object t)
		{
			if (parent != null)
			{
				((ITree)parent).ReplaceChildren(startChildIndex, stopChildIndex, t);
			}
		}
	}
}