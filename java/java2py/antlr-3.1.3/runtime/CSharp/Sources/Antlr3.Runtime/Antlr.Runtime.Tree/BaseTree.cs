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
	using IList			= System.Collections.IList;
	using ArrayList		= System.Collections.ArrayList;
	using StringBuilder = System.Text.StringBuilder;
	
	/// <summary>
	/// A generic tree implementation with no payload.  You must subclass to
	/// actually have any user data.  ANTLR v3 uses a list of children approach
	/// instead of the child-sibling approach in v2.  A flat tree (a list) is
	/// an empty node whose children represent the list.  An empty, but
	/// non-null node is called "nil".
	/// </summary>
	[Serializable]
	public abstract class BaseTree : ITree
	{
		public BaseTree()
		{
		}
		
		/// <summary>Create a new node from an existing node does nothing for BaseTree
		/// as there are no fields other than the children list, which cannot
		/// be copied as the children are not considered part of this node. 
		/// </summary>
		public BaseTree(ITree node)
		{
		}
		
		virtual public int ChildCount
		{
			get
			{
				if (children == null)
				{
					return 0;
				}
				return children.Count;
			}
		}

		virtual public bool IsNil
		{
			get { return false; }
		}

		virtual public int Line
		{
			get { return 0; }
		}

		virtual public int CharPositionInLine
		{
			get { return 0; }
		}

		protected IList children;
		
		public virtual ITree GetChild(int i)
		{
			if (children == null || i >= children.Count)
			{
				return null;
			}
			return (ITree) children[i];
		}

		/// <summary>
		/// Get the children internal list of children. Manipulating the list
		/// directly is not a supported operation (i.e. you do so at your own risk)
		/// </summary>
		public IList Children
		{
			get { return children; }
		}

		/// <summary>
		/// Add t as child of this node.
		/// </summary>
		/// <remarks>
		/// Warning: if t has no children, but child does and child isNil then 
		/// this routine moves children to t via t.children = child.children; 
		/// i.e., without copying the array.
		/// </remarks>
		/// <param name="t"></param>
		public virtual void AddChild(ITree t)
		{
			if (t == null)
			{
				return;
			}

			BaseTree childTree = (BaseTree)t;
			if (childTree.IsNil)	// t is an empty node possibly with children
			{
				if ((children != null) && (children == childTree.children))
				{
					throw new InvalidOperationException("attempt to add child list to itself");
				}
				// just add all of childTree's children to this
				if (childTree.children != null)
				{
					if (children != null) // must copy, this has children already
					{
						int n = childTree.children.Count;
						for (int i = 0; i < n; i++)
						{
							ITree c = (ITree)childTree.Children[i];
							children.Add(c);
							// handle double-link stuff for each child of nil root
							c.Parent = this;
							c.ChildIndex = children.Count - 1;
						}
					}
					else
					{
						// no children for this but t has children; just set pointer
						// call general freshener routine
						children = childTree.children;
						FreshenParentAndChildIndexes();
					}
				}
			}
			else
			{
				// child is not nil (don't care about children)
				if (children == null)
				{
					children = CreateChildrenList(); // create children list on demand
				}
				children.Add(t);
				childTree.Parent = this;
				childTree.ChildIndex = children.Count - 1;
			}
		}
		
		/// <summary>
		/// Add all elements of kids list as children of this node
		/// </summary>
		/// <param name="kids"></param>
		public void AddChildren(IList kids) 
		{
			for (int i = 0; i < kids.Count; i++) 
			{
				ITree t = (ITree) kids[i];
				AddChild(t);
			}
		}

		public virtual void SetChild(int i, ITree t)
		{
			if (t == null)
			{
				return;
			}
			if (t.IsNil)
			{
				throw new ArgumentException("Can't set single child to a list");
			}
			if (children == null)
			{
				children = CreateChildrenList();
			}
			children[i] = t;
			t.Parent = this;
			t.ChildIndex = i;
		}
		
		public virtual object DeleteChild(int i)
		{
			if (children == null)
			{
				return null;
			}
			ITree killed = (ITree)children[i];
			children.RemoveAt(i);
			// walk rest and decrement their child indexes
			FreshenParentAndChildIndexes(i);
			return killed;
		}

		/// <summary>
		/// Delete children from start to stop and replace with t even if t is
		/// a list (nil-root tree).
		/// </summary>
		/// <remarks>
		/// Number of children can increase or decrease.
		/// For huge child lists, inserting children can force walking rest of
		/// children to set their childindex; could be slow.
		/// </remarks>
		public virtual void ReplaceChildren(int startChildIndex, int stopChildIndex, object t)
		{
			/*
			Console.Out.WriteLine("replaceChildren "+startChildIndex+", "+stopChildIndex+
							   " with "+((BaseTree)t).ToStringTree());
			Console.Out.WriteLine("in="+ToStringTree());
			*/
			if (children == null)
			{
				throw new ArgumentException("indexes invalid; no children in list");
			}
			int replacingHowMany = stopChildIndex - startChildIndex + 1;
			int replacingWithHowMany;
			BaseTree newTree = (BaseTree)t;
			IList newChildren;
			// normalize to a list of children to add: newChildren
			if (newTree.IsNil)
			{
				newChildren = newTree.Children;
			}
			else
			{
				newChildren = new ArrayList(1);
				newChildren.Add(newTree);
			}
			replacingWithHowMany = newChildren.Count;
			int numNewChildren = newChildren.Count;
			int delta = replacingHowMany - replacingWithHowMany;
			// if same number of nodes, do direct replace
			if (delta == 0)
			{
				int j = 0; // index into new children
				for (int i = startChildIndex; i <= stopChildIndex; i++)
				{
					BaseTree child = (BaseTree)newChildren[j];
					children[i] = child;
					child.Parent = this;
					child.ChildIndex = i;
					j++;
				}
			}
			else if (delta > 0)
			{ // fewer new nodes than there were
				// set children and then delete extra
				for (int j = 0; j < numNewChildren; j++)
				{
					children[startChildIndex + j] = newChildren[j];
				}
				int indexToDelete = startChildIndex + numNewChildren;
				for (int c = indexToDelete; c <= stopChildIndex; c++)
				{
					// delete same index, shifting everybody down each time
					children.RemoveAt(indexToDelete);
				}
				FreshenParentAndChildIndexes(startChildIndex);
			}
			else
			{ // more new nodes than were there before
				// fill in as many children as we can (replacingHowMany) w/o moving data
				int replacedSoFar;
				for (replacedSoFar = 0; replacedSoFar < replacingHowMany; replacedSoFar++)
				{
					children[startChildIndex + replacedSoFar] = newChildren[replacedSoFar];
				}
				// replacedSoFar has correct index for children to add
				for ( ; replacedSoFar < replacingWithHowMany; replacedSoFar++)
				{
					children.Insert(startChildIndex + replacedSoFar, newChildren[replacedSoFar]);
				}
				FreshenParentAndChildIndexes(startChildIndex);
			}
			//Console.Out.WriteLine("out="+ToStringTree());
		}

		/// <summary>Override in a subclass to change the impl of children list </summary>
		protected internal virtual IList CreateChildrenList()
		{
			return new ArrayList();
		}

		/// <summary>Set the parent and child index values for all child of t</summary>
		public virtual void FreshenParentAndChildIndexes()
		{
			FreshenParentAndChildIndexes(0);
		}

		public virtual void FreshenParentAndChildIndexes(int offset)
		{
			int n = ChildCount;
			for (int c = offset; c < n; c++)
			{
				ITree child = (ITree)GetChild(c);
				child.ChildIndex	= c;
				child.Parent = this;
			}
		}

		public virtual void SanityCheckParentAndChildIndexes()
		{
			SanityCheckParentAndChildIndexes(null, -1);
		}

		public virtual void SanityCheckParentAndChildIndexes(ITree parent, int i)
		{
			if (parent != this.Parent)
			{
				throw new ArgumentException("parents don't match; expected " + parent + " found " + this.Parent);
			}
			if (i != this.ChildIndex)
			{
				throw new NotSupportedException("child indexes don't match; expected " + i + " found " + this.ChildIndex);
			}
			int n = this.ChildCount;
			for (int c = 0; c < n; c++)
			{
				CommonTree child = (CommonTree)this.GetChild(c);
				child.SanityCheckParentAndChildIndexes(this, c);
			}
		}

		/// <summary>BaseTree doesn't track child indexes.</summary>
		public virtual int ChildIndex
		{
			get { return 0; }
			set { }
		}

		/// <summary>BaseTree doesn't track parent pointers.</summary>
		public virtual ITree Parent
		{
			get { return null; }
			set { }
		}

	    /// <summary>
	    ///  Walk upwards looking for ancestor with this token type.
	    /// </summary>
	    public bool HasAncestor(int ttype) { return GetAncestor(ttype)!=null; }
	
	    /// <summary>
	    /// Walk upwards and get first ancestor with this token type.
	    /// </summary>
	    public ITree GetAncestor(int ttype) {
	        ITree t = this;
	        t = t.Parent;
	        while ( t!=null ) {
	            if ( t.Type == ttype ) return t;
	            t = t.Parent;
	        }
	        return null;
	    }
	
	    /// <summary>
	    /// Return a list of all ancestors of this node.  The first node of
	    /// list is the root and the last is the parent of this node.
	    /// </summary>
	    public IList GetAncestors() {
	        if ( Parent==null ) return null;
	        IList ancestors = new ArrayList();
	        ITree t = this;
	        t = t.Parent;
	        while ( t!=null ) {
	            ancestors.Insert(0, t); // insert at start
	            t = t.Parent;
	        }
	        return ancestors;
	    }

		/// <summary>
		/// Print out a whole tree not just a node
		/// </summary>
		public virtual string ToStringTree()
		{
			if (children == null || children.Count == 0)
			{
				return this.ToString();
			}
			StringBuilder buf = new StringBuilder();
			if (!IsNil)
			{
				buf.Append("(");
				buf.Append(this.ToString());
				buf.Append(' ');
			}
			for (int i = 0; children != null && i < children.Count; i++)
			{
				ITree t = (ITree) children[i];
				if (i > 0)
				{
					buf.Append(' ');
				}
				buf.Append(t.ToStringTree());
			}
			if (!IsNil)
			{
				buf.Append(")");
			}
			return buf.ToString();
		}
		
		/// <summary>
		/// Force base classes override and say how a node (not a tree) 
		/// should look as text 
		/// </summary>
		public override abstract string ToString();

		public abstract ITree DupNode();

		public abstract int Type
		{
			get;
		}

		public abstract int TokenStartIndex
		{
			get;
			set;
		}

		public abstract int TokenStopIndex
		{
			get;
			set;
		}

		public abstract string Text
		{
			get;
		}
	}
}