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
	using IDictionary = System.Collections.IDictionary;
	using Hashtable = System.Collections.Hashtable;
	using IToken = Antlr.Runtime.IToken;

	/// <summary>
	/// A TreeAdaptor that works with any Tree implementation
	/// </summary>
	public abstract class BaseTreeAdaptor : ITreeAdaptor
	{
		/// <summary>A map of tree node to unique IDs.</summary>
		protected IDictionary treeToUniqueIDMap;

		/// <summary>Next available unique ID.</summary>
		protected int uniqueNodeID = 1;

		public virtual object GetNilNode()
		{
			return Create(null);
		}
		
		/// <summary>
		/// Create tree node that holds the start and stop tokens associated
		///  with an error.
		/// </summary>
		/// <remarks>
		/// <para>If you specify your own kind of tree nodes, you will likely have to
		/// override this method. CommonTree returns Token.INVALID_TOKEN_TYPE
		/// if no token payload but you might have to set token type for diff
		/// node type.</para>
		///
		/// <para>You don't have to subclass CommonErrorNode; you will likely need to
		/// subclass your own tree node class to avoid class cast exception.</para>
		/// </remarks>
		public virtual object ErrorNode(ITokenStream input, IToken start, IToken stop,
								RecognitionException e)
		{
			CommonErrorNode t = new CommonErrorNode(input, start, stop, e);
			//System.out.println("returning error node '"+t+"' @index="+input.index());
			return t;
		}
	
		public virtual bool IsNil(object tree) 
		{
			return ((ITree)tree).IsNil;
		}

		public virtual object DupTree(object tree)
		{
			return DupTree(tree, null);
		}

		/// <summary>
		/// This is generic in the sense that it will work with any kind of
		/// tree (not just the ITree interface).  It invokes the adaptor routines
		/// not the tree node routines to do the construction.  
		/// </summary>
		public virtual object DupTree(object t, object parent)
		{
			if (t == null)
			{
				return null;
			}
			object newTree = DupNode(t);
			// ensure new subtree root has parent/child index set
			SetChildIndex(newTree, GetChildIndex(t)); // same index in new tree
			SetParent(newTree, parent);
			int n = GetChildCount(t);
			for (int i = 0; i < n; i++)
			{
				object child = GetChild(t, i);
				object newSubTree = DupTree(child, t);
				AddChild(newTree, newSubTree);
			}
			return newTree;
		}

		/// <summary>
		/// Add a child to the tree t.  If child is a flat tree (a list), make all
		/// in list children of t.
		/// </summary>
		/// <remarks>
		/// <para>
		/// Warning: if t has no children, but child does and child isNil 
		/// then you can decide it is ok to move children to t via 
		/// t.children = child.children; i.e., without copying the array. 
		/// Just make sure that this is consistent with how the user will build
		/// ASTs.
		///	</para>
		/// </remarks>
		public virtual void  AddChild(object t, object child)
		{
			if ((t != null) && (child != null))
			{
				((ITree) t).AddChild((ITree) child);
			}
		}
		
		/// <summary>
		/// If oldRoot is a nil root, just copy or move the children to newRoot.
		/// If not a nil root, make oldRoot a child of newRoot.
		/// </summary>
		/// <remarks>
		/// 
		///   old=^(nil a b c), new=r yields ^(r a b c)
		///   old=^(a b c), new=r yields ^(r ^(a b c))
		/// 
		/// If newRoot is a nil-rooted single child tree, use the single
		/// child as the new root node.
		/// 
		///   old=^(nil a b c), new=^(nil r) yields ^(r a b c)
		///   old=^(a b c), new=^(nil r) yields ^(r ^(a b c))
		///  
		/// If oldRoot was null, it's ok, just return newRoot (even if isNil).
		/// 
		///   old=null, new=r yields r
		///   old=null, new=^(nil r) yields ^(nil r)
		/// 
		/// Return newRoot.  Throw an exception if newRoot is not a
		/// simple node or nil root with a single child node--it must be a root
		/// node.  If newRoot is ^(nil x) return x as newRoot.
		/// 
		/// Be advised that it's ok for newRoot to point at oldRoot's
		/// children; i.e., you don't have to copy the list.  We are
		/// constructing these nodes so we should have this control for
		/// efficiency.
		/// </remarks>
		public virtual object BecomeRoot(object newRoot, object oldRoot)
		{
			ITree newRootTree = (ITree) newRoot;
			ITree oldRootTree = (ITree) oldRoot;
			if (oldRoot == null)
			{
				return newRoot;
			}
			// handle ^(nil real-node)
			if (newRootTree.IsNil)
			{
	            int nc = newRootTree.ChildCount;
	            if ( nc==1 ) newRootTree = (ITree)newRootTree.GetChild(0);
	            else if ( nc >1 ) {
					throw new SystemException("more than one node as root (TODO: make exception hierarchy)");
				}
			}
			// add oldRoot to newRoot; AddChild takes care of case where oldRoot
			// is a flat list (i.e., nil-rooted tree).  All children of oldRoot
			// are added to newRoot.
			newRootTree.AddChild(oldRootTree);
			return newRootTree;
		}

		/// <summary>Transform ^(nil x) to x and nil to null</summary>
		public virtual object RulePostProcessing(object root)
		{
			ITree r = (ITree) root;
			if (r != null && r.IsNil)
			{
				if (r.ChildCount == 0)
				{
					r = null;
				}
				else if (r.ChildCount == 1)
				{
					r = (ITree)r.GetChild(0);
					// whoever invokes rule will set parent and child index
					r.Parent = null;
					r.ChildIndex = -1;
				}
			}
			return r;
		}
		
		public virtual object BecomeRoot(IToken newRoot, object oldRoot)
		{
			return BecomeRoot(Create(newRoot), oldRoot);
		}
		
		public virtual object Create(int tokenType, IToken fromToken)
		{
			fromToken = CreateToken(fromToken);
			fromToken.Type = tokenType;
			ITree t = (ITree) Create(fromToken);
			return t;
		}
		
		public virtual object Create(int tokenType, IToken fromToken, string text)
		{
			fromToken = CreateToken(fromToken);
			fromToken.Type = tokenType;
			fromToken.Text = text;
			ITree t = (ITree) Create(fromToken);
			return t;
		}
		
		public virtual object Create(int tokenType, string text)
		{
			IToken fromToken = CreateToken(tokenType, text);
			ITree t = (ITree) Create(fromToken);
			return t;
		}
		
		public virtual int GetNodeType(object t)
		{
			return ((ITree) t).Type;
		}

		public virtual void SetNodeType(object t, int type)
		{
			throw new NotImplementedException("don't know enough about Tree node");
		}

		public virtual string GetNodeText(object t)
		{
			return ((ITree)t).Text;
		}

		public virtual void SetNodeText(object t, string text)
		{
			throw new NotImplementedException("don't know enough about Tree node");
		}
		
		public virtual object GetChild(object t, int i)
		{
			return ((ITree)t).GetChild(i);
		}

		public virtual void SetChild(object t, int i, object child)
		{
			((ITree)t).SetChild(i, (ITree)child);
		}

		public virtual object DeleteChild(object t, int i)
		{
			return ((ITree)t).DeleteChild(i);
		}

		public virtual int GetChildCount(object t)
		{
			return ((ITree)t).ChildCount;
			
		}

		public abstract object DupNode(object param1);
		public abstract object Create(IToken param1);
		public abstract void  SetTokenBoundaries(object param1, IToken param2, IToken param3);
		public abstract int GetTokenStartIndex(object t);
		public abstract int GetTokenStopIndex(object t);
		public abstract IToken GetToken(object treeNode);

		/// <summary>
		/// For identifying trees. How to identify nodes so we can say "add node 
		/// to a prior node"?
		/// </summary>
		/// <remarks>
		/// <para>
		/// System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode() is 
		/// not available in .NET 1.0. It is "broken/buggy" in .NET 1.1 
		/// (for multi-appdomain scenarios).
		/// </para>
		/// <para>
		/// We are tracking uniqueness of IDs ourselves manually since ANTLR 
		/// v3.1 release using hashtables. We will be tracking . Even though 
		/// it is expensive, we will create a hashtable with all tree nodes 
		/// in it as this is only for debugging. 
		/// </para>
		/// </remarks>
		public int GetUniqueID(object node)
		{
			if (treeToUniqueIDMap == null)
			{
				treeToUniqueIDMap = new Hashtable();
			}
			object prevIdObj = treeToUniqueIDMap[node];
			if (prevIdObj != null)
			{
				return (int) prevIdObj;
			}
			int ID = uniqueNodeID;
			treeToUniqueIDMap[node] = ID;
			uniqueNodeID++;
			return ID;

			//return node.GetHashCode();
			//return System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(node);
		}

		/// <summary>
		/// Tell me how to create a token for use with imaginary token nodes.
		/// For example, there is probably no input symbol associated with imaginary
		/// token DECL, but you need to create it as a payload or whatever for
		/// the DECL node as in ^(DECL type ID).
		/// 
		/// If you care what the token payload objects' type is, you should
		/// override this method and any other createToken variant.
		/// </summary>
		public abstract IToken CreateToken(int tokenType, string text);

		/// <summary>
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
		/// </summary>
		public abstract IToken CreateToken(IToken fromToken);

		/// <summary>
		/// Who is the parent node of this node; if null, implies node is root.
		/// </summary>
		/// <remarks>
		/// If your node type doesn't handle this, it's ok but the tree rewrites
		/// in tree parsers need this functionality.
		/// </remarks>
		public abstract object GetParent(object t);
		public abstract void SetParent(object t, object parent);

		/// <summary>
		/// What index is this node in the child list? Range: 0..n-1
		/// </summary>
		/// <remarks>
		/// If your node type doesn't handle this, it's ok but the tree rewrites
		/// in tree parsers need this functionality.
		/// </remarks>
		public abstract int GetChildIndex(object t);
		public abstract void SetChildIndex(object t, int index);

		/// <summary>
		/// Replace from start to stop child index of parent with t, which might
		/// be a list.  Number of children may be different after this call.
		/// </summary>
		/// <remarks>
		/// If parent is null, don't do anything; must be at root of overall tree.
		/// Can't replace whatever points to the parent externally.  Do nothing.
		/// </remarks>
		public abstract void ReplaceChildren(object parent, int startChildIndex, int stopChildIndex, object t);
	}
}