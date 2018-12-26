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
	using StringBuilder	= System.Text.StringBuilder;
	using IList			= System.Collections.IList;
	using ArrayList		= System.Collections.ArrayList;
	using StackList		= Antlr.Runtime.Collections.StackList;
	using IToken		= Antlr.Runtime.IToken;
	using ITokenStream	= Antlr.Runtime.ITokenStream;
	
	/// <summary>
	/// A stream of tree nodes, accessing nodes from a tree of ANY kind.
	/// </summary>
	/// <remarks>
	/// No new nodes should be created in tree during the walk.  A small buffer
	/// of tokens is kept to efficiently and easily handle LT(i) calls, though
	/// the lookahead mechanism is fairly complicated.
	/// 
	/// For tree rewriting during tree parsing, this must also be able
	/// to replace a set of children without "losing its place".
	/// That part is not yet implemented.  Will permit a rule to return
	/// a different tree and have it stitched into the output tree probably.
	/// 
	/// <see cref="CommonTreeNodeStream"/>
	/// 
	/// </remarks>
	public class UnBufferedTreeNodeStream : ITreeNodeStream
	{
		/// <summary>
		/// Where is this stream pulling nodes from?  This is not the name, but
		/// the object that provides node objects.
		/// </summary>
		virtual public object TreeSource
		{
			get { return root; }
		}

		#region IEnumerator Members (implementation)

		private ITree currentEnumerationNode;

		virtual public void  Reset()
		{
			currentNode = root;
			previousNode = null;
			currentChildIndex = - 1;
			absoluteNodeIndex = -1;
			head = tail = 0;
		}

		/// <summary>
		/// Navigates to the next node found during a depth-first walk of root.
		/// Also, adds these nodes and DOWN/UP imaginary nodes into the lokoahead
		/// buffer as a side-effect.  Normally side-effects are bad, but because
		/// we can Emit many tokens for every MoveNext() call, it's pretty hard to
		/// use a single return value for that.  We must add these tokens to
		/// the lookahead buffer.
		/// 
		/// This routine does *not* cause the 'Current' property to ever return the 
		/// DOWN/UP nodes; those are only returned by the LT() method.
		/// 
		/// Ugh.  This mechanism is much more complicated than a recursive
		/// solution, but it's the only way to provide nodes on-demand instead
		/// of walking once completely through and buffering up the nodes. :(
		/// </summary>
		public virtual bool MoveNext()
		{
			// already walked entire tree; nothing to return
			if (currentNode == null)
			{
				AddLookahead(eof);
				currentEnumerationNode = null;
				// this is infinite stream returning EOF at end forever
				// so don't throw NoSuchElementException
				return false;
			}
				
			// initial condition (first time method is called)
			if (currentChildIndex == - 1)
			{
				currentEnumerationNode = (ITree)handleRootNode();
				return true;
			}
				
			// index is in the child list?
			if (currentChildIndex < adaptor.GetChildCount(currentNode))
			{
				currentEnumerationNode = (ITree)VisitChild(currentChildIndex);
				return true;
			}
				
			// hit end of child list, return to parent node or its parent ...
			WalkBackToMostRecentNodeWithUnvisitedChildren();
			if (currentNode != null)
			{
				currentEnumerationNode = (ITree)VisitChild(currentChildIndex);
				return true;
			}
			return false;
		}
		
		public virtual object Current
		{
			get { return currentEnumerationNode; }
		}

		#endregion

		public const int INITIAL_LOOKAHEAD_BUFFER_SIZE = 5;

		/// <summary>Reuse same DOWN, UP navigation nodes unless this is true</summary>
		protected bool uniqueNavigationNodes = false;

		/// <summary>Pull nodes from which tree? </summary>
		protected internal object root;

		/// <summary>IF this tree (root) was created from a token stream, track it.</summary>
		protected ITokenStream tokens;

		/// <summary>What tree adaptor was used to build these trees</summary>
		ITreeAdaptor adaptor;

		/// <summary>
		/// As we walk down the nodes, we must track parent nodes so we know
		/// where to go after walking the last child of a node.  When visiting
		/// a child, push current node and current index.
		/// </summary>
		protected internal StackList nodeStack = new StackList();
		
		/// <summary>
		/// Track which child index you are visiting for each node we push.
		/// TODO: pretty inefficient...use int[] when you have time
		/// </summary>
		protected internal StackList indexStack = new StackList();

		/// <summary>Which node are we currently visiting? </summary>
		protected internal object currentNode;
		
		/// <summary>Which node did we visit last?  Used for LT(-1) calls. </summary>
		protected internal object previousNode;
		
		/// <summary>
		/// Which child are we currently visiting?  If -1 we have not visited
		/// this node yet; next Consume() request will set currentIndex to 0.
		/// </summary>
		protected internal int currentChildIndex;

		/// <summary>
		/// What node index did we just consume?  i=0..n-1 for n node trees.
		/// IntStream.next is hence 1 + this value.  Size will be same.
		/// </summary>
		protected int absoluteNodeIndex;

		/// <summary>
		/// Buffer tree node stream for use with LT(i).  This list grows
		/// to fit new lookahead depths, but Consume() wraps like a circular
		/// buffer.
		/// </summary>
		protected internal object[] lookahead = new object[INITIAL_LOOKAHEAD_BUFFER_SIZE];
		
		/// <summary>lookahead[head] is the first symbol of lookahead, LT(1). </summary>
		protected internal int head;
		
		/// <summary>
		/// Add new lookahead at lookahead[tail].  tail wraps around at the
		/// end of the lookahead buffer so tail could be less than head.
		/// </summary>
		protected internal int tail;

		/// <summary>
		/// When walking ahead with cyclic DFA or for syntactic predicates,
		/// we need to record the state of the tree node stream.  This
		/// class wraps up the current state of the UnBufferedTreeNodeStream.
		/// Calling Mark() will push another of these on the markers stack.
		/// </summary>
		protected class TreeWalkState
		{
			protected internal int currentChildIndex;
			protected internal int absoluteNodeIndex;
			protected internal object currentNode;
			protected internal object previousNode;
			///<summary>Record state of the nodeStack</summary>
			protected internal int nodeStackSize;
			///<summary>Record state of the indexStack</summary>
			protected internal int indexStackSize;
			protected internal object[] lookahead;
		}

		/// <summary>
		/// Calls to Mark() may be nested so we have to track a stack of them.
		/// The marker is an index into this stack. This is a List&lt;TreeWalkState&gt;.
		/// Indexed from 1..markDepth. A null is kept at index 0. It is created
		/// upon first call to Mark().
		/// </summary>
		protected IList markers;

		///<summary>
		/// tracks how deep Mark() calls are nested
		/// </summary>
		protected int markDepth = 0;

		///<summary>
		/// Track the last Mark() call result value for use in Rewind().
		/// </summary>
		protected int lastMarker;

		// navigation nodes

		protected object down;
		protected object up;
		protected object eof;

		public UnBufferedTreeNodeStream(object tree)
			: this(new CommonTreeAdaptor(), tree)
		{
		}

		public UnBufferedTreeNodeStream(ITreeAdaptor adaptor, object tree) 
		{
			this.root = tree;
			this.adaptor = adaptor;
			Reset();
			down = adaptor.Create(Token.DOWN, "DOWN");
			up = adaptor.Create(Token.UP, "UP");
			eof = adaptor.Create(Token.EOF, "EOF");
		}


		public virtual object Get(int i) 
		{
			throw new NotSupportedException("stream is unbuffered");
		}

		/// <summary>
		/// Get tree node at current input pointer + i ahead where i=1 is next node.
		/// i &lt; 0 indicates nodes in the past.  So -1 is previous node and -2 is
		/// two nodes ago. LT(0) is undefined.  For i>=n, return null.
		/// Return null for LT(0) and any index that results in an absolute address
		/// that is negative.
		/// 
		/// This is analogus to the LT() method of the TokenStream, but this
		/// returns a tree node instead of a token.  Makes code gen identical
		/// for both parser and tree grammars. :)
		/// </summary>
		public virtual object LT(int k)
		{
			if (k == - 1)
			{
				return previousNode;
			}
			if (k < 0)
			{
				throw new ArgumentNullException("tree node streams cannot look backwards more than 1 node", "k");
			}
			if (k == 0)
			{
				return Tree.INVALID_NODE;
			}
			fill(k);
			return lookahead[(head + k - 1) % lookahead.Length];
		}
		
		/// <summary>Make sure we have at least k symbols in lookahead buffer </summary>
		protected internal virtual void  fill(int k)
		{
			int n = LookaheadSize;
			for (int i = 1; i <= k - n; i++)
			{
				MoveNext(); // get at least k-depth lookahead nodes
			}
		}
		
		/// <summary>
		/// Add a node to the lookahead buffer.  Add at lookahead[tail].
		/// If you tail+1 == head, then we must create a bigger buffer
		/// and copy all the nodes over plus reset head, tail.  After
		/// this method, LT(1) will be lookahead[0].
		/// </summary>
		protected internal virtual void AddLookahead(object node)
		{
			lookahead[tail] = node;
			tail = (tail + 1) % lookahead.Length;
			if (tail == head)
			{
				// buffer overflow: tail caught up with head
				// allocate a buffer 2x as big
				object[] bigger = new object[2 * lookahead.Length];
				// copy head to end of buffer to beginning of bigger buffer
				int remainderHeadToEnd = lookahead.Length - head;
				Array.Copy(lookahead, head, bigger, 0, remainderHeadToEnd);
				// copy 0..tail to after that
				Array.Copy(lookahead, 0, bigger, remainderHeadToEnd, tail);
				lookahead = bigger; // reset to bigger buffer
				head = 0;
				tail += remainderHeadToEnd;
			}
		}
		
		// Satisfy IntStream interface
		
		public virtual void  Consume()
		{
			// make sure there is something in lookahead buf, which might call next()
			fill(1);
			absoluteNodeIndex++;
			previousNode = lookahead[head]; // track previous node before moving on
			head = (head + 1) % lookahead.Length;
		}
		
		public virtual int LA(int i)
		{
			object t = (ITree) LT(i);
			if (t == null)
			{
				return Token.INVALID_TOKEN_TYPE;
			}
			return adaptor.GetNodeType(t);
		}

		/// <summary>
		/// Record the current state of the tree walk which includes
		/// the current node and stack state as well as the lookahead
		/// buffer.
		/// </summary>
		public virtual int Mark()
		{
			if (markers == null)
			{
				markers = new ArrayList();
				markers.Add(null); // depth 0 means no backtracking, leave blank
			}
			markDepth++;
			TreeWalkState state = null;
			if ( markDepth >= markers.Count ) 
			{
				state = new TreeWalkState();
				markers.Add(state);
			}
			else 
			{
				state = (TreeWalkState)markers[markDepth];

			}
			state.absoluteNodeIndex = absoluteNodeIndex;
			state.currentChildIndex = currentChildIndex;
			state.currentNode = currentNode;
			state.previousNode = previousNode;
			state.nodeStackSize = nodeStack.Count;
			state.indexStackSize = indexStack.Count;
			// take snapshot of lookahead buffer
			int n = LookaheadSize;
			int i = 0;
			state.lookahead = new object[n];
			for (int k = 1; k <= n; k++, i++)
			{
				state.lookahead[i] = LT(k);
			}
			lastMarker = markDepth;
			return markDepth;
		}
		
		public virtual void  Release(int marker)
		{
			// unwind any other markers made after marker and release marker
			markDepth = marker;
			// release this marker
			markDepth--;
		}
		
		/// <summary>
		/// Rewind the current state of the tree walk to the state it
		/// was in when Mark() was called and it returned marker.  Also,
		/// wipe out the lookahead which will force reloading a few nodes
		/// but it is better than making a copy of the lookahead buffer
		/// upon Mark().
		/// </summary>
		public virtual void Rewind(int marker)
		{
			if ( markers == null ) 
			{
				return;
			}
			TreeWalkState state = (TreeWalkState)markers[marker];
			absoluteNodeIndex = state.absoluteNodeIndex;
			currentChildIndex = state.currentChildIndex;
			currentNode = state.currentNode;
			previousNode = state.previousNode;
			// drop node and index stacks back to old size
			nodeStack.Capacity = state.nodeStackSize;
			indexStack.Capacity = state.indexStackSize;
			head = tail = 0; // wack lookahead buffer and then refill
			for (; tail < state.lookahead.Length; tail++)
			{
				lookahead[tail] = state.lookahead[tail];
			}
			Release(marker);
		}

		public void Rewind()
		{
			Rewind(lastMarker);
		}

		/// <summary>
		/// Consume() ahead until we hit index.  Can't just jump ahead--must
		/// spit out the navigation nodes.
		/// </summary>
		public virtual void Seek(int index)
		{
			if (index < this.Index())
			{
				throw new ArgumentOutOfRangeException("can't seek backwards in node stream", "index");
			}
			// seek forward, consume until we hit index
			while (this.Index() < index)
			{
				Consume();
			}
		}

		public virtual int Index()
		{
			return absoluteNodeIndex + 1;
		}
		
		/// <summary>
		/// Expensive to compute; recursively walk tree to find size;
		/// include navigation nodes and EOF.  Reuse functionality
		/// in CommonTreeNodeStream as we only really use this
		/// for testing.
		/// </summary>
		[Obsolete("Please use property Count instead.")]
		public virtual int Size()
		{
			return Count;

		}
		
		/// <summary>
		/// Expensive to compute; recursively walk tree to find size;
		/// include navigation nodes and EOF.  Reuse functionality
		/// in CommonTreeNodeStream as we only really use this
		/// for testing.
		/// </summary>
		public virtual int Count
		{
			get
			{
				CommonTreeNodeStream s = new CommonTreeNodeStream(root);
				return s.Count;
			}
		}
		
		// Satisfy Java's Iterator interface
		
		protected internal virtual object handleRootNode()
		{
			object node;
			node = currentNode;
			// point to first child in prep for subsequent next()
			currentChildIndex = 0;
			if ( adaptor.IsNil(node) )
			{
				// don't count this root nil node
				node = VisitChild(currentChildIndex);
			}
			else
			{
				AddLookahead(node);
				if ( adaptor.GetChildCount(currentNode) == 0 )
				{
					// single node case
					currentNode = null; // say we're done
				}
			}
			return node;
		}
		
		protected internal virtual object VisitChild(int child)
		{
			object node = null;
			// save state
			nodeStack.Push(currentNode);
			indexStack.Push(child);
			if (child == 0 && !adaptor.IsNil(currentNode))
			{
				AddNavigationNode(Token.DOWN);
			}
			// visit child
			currentNode = adaptor.GetChild(currentNode, child);
			currentChildIndex = 0;
			node = currentNode; // record node to return
			AddLookahead(node);
			WalkBackToMostRecentNodeWithUnvisitedChildren();
			return node;
		}

		/// <summary>
		/// As we flatten the tree, we use UP, DOWN nodes to represent
		/// the tree structure.  When debugging we need unique nodes
		/// so instantiate new ones when uniqueNavigationNodes is true.
		/// </summary>
		protected internal virtual void AddNavigationNode(int ttype)
		{
			object navNode = null;
			if (ttype == Token.DOWN)
			{
				if (HasUniqueNavigationNodes) 
					navNode = adaptor.Create(Token.DOWN, "DOWN");
				else 
					navNode = down;
			}
			else
			{
				if ( HasUniqueNavigationNodes ) 
				{
					navNode = adaptor.Create(Token.UP, "UP");
				}
				else 
				{
					navNode = up;
				}
			}
			AddLookahead(navNode);
		}

		/// <summary>
		///  Walk upwards looking for a node with more children to walk.
		/// </summary>
		protected internal virtual void WalkBackToMostRecentNodeWithUnvisitedChildren()
		{
			while ( (currentNode != null) && (currentChildIndex >= adaptor.GetChildCount(currentNode)) )
			{
				currentNode = nodeStack.Pop();
				if ( currentNode == null ) 
				{ // hit the root?
					return;
				}
				currentChildIndex = ((int) indexStack.Pop());
				currentChildIndex++; // move to next child
				if (currentChildIndex >= adaptor.GetChildCount(currentNode))
				{
					if ( !adaptor.IsNil(currentNode) )
					{
						AddNavigationNode(Token.UP);
					}
					if (currentNode == root)
					{
						// we done yet?
						currentNode = null;
					}
				}
			}
		}

		public ITreeAdaptor TreeAdaptor
		{
			get { return adaptor; }
		}

		public string SourceName {
			get { return TokenStream.SourceName; }
		}

		public ITokenStream TokenStream
		{
			get { return tokens;  }
			set { tokens = value; }
		}

		public bool HasUniqueNavigationNodes
		{
			get { return uniqueNavigationNodes;  }
			set { uniqueNavigationNodes = value; }
		}

		public void ReplaceChildren(object parent, int startChildIndex, int stopChildIndex, object t)
		{
			throw new NotSupportedException("can't do stream rewrites yet");
		}

		/// <summary>
		/// Print out the entire tree including DOWN/UP nodes.  Uses
		/// a recursive walk.  Mostly useful for testing as it yields
		/// the token types not text.
		/// </summary>
		public override string ToString()
		{
			return ToString(root, null);
		}

		protected int LookaheadSize
		{
			get { return tail < head ? (lookahead.Length - head + tail) : (tail - head); }
		}

		/// <summary>TODO: not sure this is what we want for trees. </summary>
		public virtual string ToString(object start, object stop)
		{
			if ( start == null ) 
			{
				return null;
			}
			// if we have the token stream, use that to dump text in order
			if ( tokens != null ) 
			{
				// don't trust stop node as it's often an UP node etc...
				// walk backwards until you find a non-UP, non-DOWN node
				// and ask for it's token index.
				int beginTokenIndex = adaptor.GetTokenStartIndex(start);
				int endTokenIndex = adaptor.GetTokenStopIndex(stop);
				if ( (stop != null) && (adaptor.GetNodeType(stop) == Token.UP) ) 
				{
					endTokenIndex = adaptor.GetTokenStopIndex(start);
				}
				else 
				{
					endTokenIndex = Count-1;
				}
				return tokens.ToString(beginTokenIndex, endTokenIndex);
			}
			StringBuilder buf = new StringBuilder();
			ToStringWork(start, stop, buf);
			return buf.ToString();
		}
		
		protected internal virtual void ToStringWork(object p, object stop, StringBuilder buf)
		{
			if ( !adaptor.IsNil(p) )
			{
				string text = adaptor.GetNodeText(p);
				if (text == null)
				{
					text = " " + adaptor.GetNodeType(p);
				}
				buf.Append(text); // ask the node to go to string
			}
			if (p == stop)
			{
				return;
			}
			int n = adaptor.GetChildCount(p);
			if ( (n > 0) && !adaptor.IsNil(p) )
			{
				buf.Append(" ");
				buf.Append(Token.DOWN);
			}
			for (int c = 0; c < n; c++)
			{
				object child = adaptor.GetChild(p, c);
				ToStringWork(child, stop, buf);
			}
			if ( (n > 0) && !adaptor.IsNil(p) )
			{
				buf.Append(" ");
				buf.Append(Token.UP);
			}
		}
	}
}