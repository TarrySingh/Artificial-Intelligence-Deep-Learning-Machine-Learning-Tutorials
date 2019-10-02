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
	using IEnumerable	= System.Collections.IEnumerable;
	using IEnumerator	= System.Collections.IEnumerator;
	using IDictionary	= System.Collections.IDictionary;
	using IList			= System.Collections.IList;
	using ArrayList		= System.Collections.ArrayList;
	using Hashtable		= System.Collections.Hashtable;
	using StringBuilder = System.Text.StringBuilder;
	using StackList		= Antlr.Runtime.Collections.StackList;
	using Token			= Antlr.Runtime.Token;
	using ITokenStream	= Antlr.Runtime.ITokenStream;
	
	/// <summary>
	/// A buffered stream of tree nodes.  Nodes can be from a tree of ANY kind.
	/// </summary>
	/// <remarks>
	/// This node stream sucks all nodes out of the tree specified in the 
	/// constructor during construction and makes pointers into the tree 
	/// using an array of Object pointers. The stream necessarily includes 
	/// pointers to DOWN and UP and EOF nodes.
	/// 
	/// This stream knows how to mark/release for backtracking.
	/// 
	/// This stream is most suitable for tree interpreters that need to
	/// jump around a lot or for tree parsers requiring speed (at cost of memory).
	/// There is some duplicated functionality here with UnBufferedTreeNodeStream
	/// but just in bookkeeping, not tree walking etc...
	/// 
	/// <see cref="UnBufferedTreeNodeStream"/>
	/// 
	/// </remarks>
	public class CommonTreeNodeStream : ITreeNodeStream, IEnumerable
	{
		public const int DEFAULT_INITIAL_BUFFER_SIZE = 100;
		public const int INITIAL_CALL_STACK_SIZE = 10;

		#region Helper classes
		protected sealed class CommonTreeNodeStreamEnumerator : IEnumerator
		{
			private CommonTreeNodeStream _nodeStream;
			private int _index;
			private object _currentItem;
		
			#region Constructors

			internal CommonTreeNodeStreamEnumerator()
			{	
			}

			internal CommonTreeNodeStreamEnumerator(CommonTreeNodeStream nodeStream)
			{
				_nodeStream = nodeStream;
				Reset();
			}

			#endregion

			#region IEnumerator Members

			public void Reset()
			{
//				if (_version != _hashList._version)
//				{
//					throw new InvalidOperationException("Collection was modified; enumeration operation may not execute.");
//				}
				_index = 0;
				_currentItem = null;
			}

			public object Current
			{
				get
				{
					if (_currentItem == null)
					{
						throw new InvalidOperationException("Enumeration has either not started or has already finished.");
					}
					return _currentItem;
				}
			}

			public bool MoveNext()
			{
//				if (_version != _hashList._version)
//				{
//					throw new InvalidOperationException("Collection was modified; enumeration operation may not execute.");
//				}

				if ( _index >= _nodeStream.nodes.Count )
				{
					int current = _index;
					_index++;
					if ( current < _nodeStream.nodes.Count ) 
					{
						_currentItem = _nodeStream.nodes[current];
					}
					_currentItem = _nodeStream.eof;
					return true;
				}
				_currentItem = null;
				return false;
			}

			#endregion
		}

		#endregion

		#region IEnumerable Members

		public IEnumerator GetEnumerator()
		{
			if ( p == -1 ) 
			{
				FillBuffer();
			}
			return new CommonTreeNodeStreamEnumerator(this);
		}

		#endregion

		#region Data Members
		// all these navigation nodes are shared and hence they
		// cannot contain any line/column info

		protected object down;
		protected object up;
		protected object eof;

		/// <summary>
		/// The complete mapping from stream index to tree node. This buffer 
		/// includes pointers to DOWN, UP, and EOF nodes.
		/// 
		/// It is built upon ctor invocation.  The elements are type Object 
		/// as we don't what the trees look like. Load upon first need of 
		/// the buffer so we can set token types of interest for reverseIndexing.
		/// Slows us down a wee bit  to do all of the if p==-1 testing everywhere though.
		/// </summary>
		protected IList nodes;

		/// <summary>Pull nodes from which tree? </summary>
		protected internal object root;

		/// <summary>IF this tree (root) was created from a token stream, track it</summary>
		protected ITokenStream tokens;

		/// <summary>What tree adaptor was used to build these trees</summary>
		ITreeAdaptor adaptor;

		/// <summary>
		/// Reuse same DOWN, UP navigation nodes unless this is true
		/// </summary>
		protected bool uniqueNavigationNodes = false;

		/// <summary>
		/// The index into the nodes list of the current node (next node
		/// to consume).  If -1, nodes array not filled yet.
		/// </summary>
		protected int p = -1;

		/// <summary>
		/// Track the last mark() call result value for use in rewind().
		/// </summary>
		protected int lastMarker;

		/// <summary>
		/// Stack of indexes used for push/pop calls
		/// </summary>
		protected StackList calls;

		#endregion

		#region Constructors

		public CommonTreeNodeStream(object tree) 
			: this(new CommonTreeAdaptor(), tree)
		{
		}

		public CommonTreeNodeStream(ITreeAdaptor adaptor, object tree) 
			: this(adaptor, tree, DEFAULT_INITIAL_BUFFER_SIZE)
		{
		}

		public CommonTreeNodeStream(ITreeAdaptor adaptor, object tree, int initialBufferSize) 
		{
			this.root = tree;
			this.adaptor = adaptor;
			nodes = new ArrayList(initialBufferSize);
			down = adaptor.Create(Token.DOWN, "DOWN");
			up = adaptor.Create(Token.UP, "UP");
			eof = adaptor.Create(Token.EOF, "EOF");
		}

		#endregion

		#region Public API

		/// <summary>
		/// Walk tree with depth-first-search and fill nodes buffer.
		/// Don't do DOWN, UP nodes if its a list (t is isNil).
		/// </summary>
		protected void FillBuffer() 
		{
			FillBuffer(root);
			p = 0; // buffer of nodes intialized now
		}

		public void FillBuffer(object t) 
		{
			bool nil = adaptor.IsNil(t);
			if ( !nil ) 
			{
				nodes.Add(t); // add this node
			}
			// add DOWN node if t has children
			int n = adaptor.GetChildCount(t);
			if ( !nil && (n > 0) ) 
			{
				AddNavigationNode(Token.DOWN);
			}
			// and now add all its children
			for (int c = 0; c < n; c++) 
			{
				object child = adaptor.GetChild(t, c);
				FillBuffer(child);
			}
			// add UP node if t has children
			if ( !nil && (n > 0) ) 
			{
				AddNavigationNode(Token.UP);
			}
		}

		/// <summary>
		/// Returns the stream index for the spcified node in the range 0..n-1 or, 
		/// -1 if node not found.
		/// </summary>
		protected int GetNodeIndex(object node)
		{
			if ( p == -1 )
			{
				FillBuffer();
			}
			for (int i = 0; i < nodes.Count; i++)
			{
				object t = (object) nodes[i];
				if ( t == node )
				{
					return i;
				}
			}
			return -1;
		}

		/// <summary>
		/// As we flatten the tree, we use UP, DOWN nodes to represent
		/// the tree structure.  When debugging we need unique nodes
		/// so instantiate new ones when uniqueNavigationNodes is true.
		/// </summary>
		protected void AddNavigationNode(int ttype)
		{
			object navNode = null;
			if ( ttype == Token.DOWN )
			{
				if ( HasUniqueNavigationNodes )
				{
					navNode = adaptor.Create(Token.DOWN, "DOWN");
				}
				else
				{
					navNode = down;
				}
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
			nodes.Add(navNode);
		}

		public object Get(int i)
		{
			if ( p == -1 )
			{
				FillBuffer();
			}
			return nodes[i];
		}

		public object LT(int k)
		{
			if ( p == -1 )
			{
				FillBuffer();
			}
			if ( k == 0 )
			{
				return null;
			}
			if ( k < 0 )
			{
				return LB(-k);
			}
			if ( (p+k-1) >= nodes.Count )
			{
				return eof;
			}
			return nodes[p+k-1];
		}

		public virtual object CurrentSymbol {
			get { return LT(1); }
		}

		/// <summary>
		/// Look backwards k nodes
		/// </summary>
		protected object LB(int k) 
		{
			if ( k == 0 )
			{
				return null;
			}
			if ( (p-k) < 0 )
			{
				return null;
			}
			return nodes[p-k];
		}

		/// <summary>
		/// Where is this stream pulling nodes from?  This is not the name, but
		/// the object that provides node objects.
		/// </summary>
		virtual public object TreeSource
		{
			get { return root; }
		}

		virtual public string SourceName {
			get { return TokenStream.SourceName; }
		}

		virtual public ITokenStream TokenStream
		{
			get { return tokens; }
			set { this.tokens = value; }
		}

		public ITreeAdaptor TreeAdaptor
		{
			get { return adaptor; }
			set { adaptor = value; }
		}

		public bool HasUniqueNavigationNodes
		{
			get { return uniqueNavigationNodes;  }
			set { uniqueNavigationNodes = value; }
		}

		/// <summary>
		/// Make stream jump to a new location, saving old location.
		/// Switch back with pop().
		/// </summary>
		public void Push(int index) 
		{
			if ( calls == null ) 
			{
				calls = new StackList();
			}
			calls.Push(p); // save current index
			Seek(index);
		}

		/// <summary>
		/// Seek back to previous index saved during last Push() call.
		/// Return top of stack (return index).
		/// </summary>
		public int Pop() 
		{
			int ret = (int)calls.Pop();
			Seek(ret);
			return ret;
		}

		public void Reset()
		{
			p = -1;
			lastMarker = 0;
			if (calls != null)
			{
				calls.Clear();
			}
		}

		#endregion

		#region Tree Rewrite Interface

		public void ReplaceChildren(object parent, int startChildIndex, int stopChildIndex, object t)
		{
			if (parent != null)
			{
				adaptor.ReplaceChildren(parent, startChildIndex, stopChildIndex, t);
			}
		}

		#endregion

		#region Satisfy IntStream interface

		public virtual void Consume()
		{
			if (p == -1)
			{
				FillBuffer();
			}
			p++;
		}
		
		public virtual int LA(int i)
		{
			return adaptor.GetNodeType( LT(i) );
		}

		/// <summary>
		/// Record the current state of the tree walk which includes
		/// the current node and stack state.
		/// </summary>
		public virtual int Mark()
		{
			if ( p == -1 ) 
			{
				FillBuffer();
			}
			lastMarker = Index();
			return lastMarker;
		}
		
		public virtual void  Release(int marker)
		{
			// no resources to release
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
			Seek(marker);
		}

		public void Rewind()
		{
			Seek(lastMarker);
		}

		/// <summary>
		/// Consume() ahead until we hit index.  Can't just jump ahead--must
		/// spit out the navigation nodes.
		/// </summary>
		public virtual void Seek(int index)
		{
			if ( p == -1 ) 
			{
				FillBuffer();
			}
			p = index;
		}

		public virtual int Index()
		{
			return p;
		}
		
		/// <summary>
		/// Expensive to compute so I won't bother doing the right thing.
		/// This method only returns how much input has been seen so far.  So
		/// after parsing it returns true size.
		/// </summary>
		[Obsolete("Please use property Count instead.")]
		public virtual int Size()
		{
			return Count;
		}

		/// <summary>
		/// Expensive to compute so I won't bother doing the right thing.
		/// This method only returns how much input has been seen so far.  So
		/// after parsing it returns true size.
		/// </summary>
		public virtual int Count
		{
			get
			{
				if ( p == -1 ) 
				{
					FillBuffer();
				}
				return nodes.Count;
			}
		}
		
		#endregion

		/// <summary>
		/// Used for testing, just return the token type stream
		/// </summary>
		public override string ToString()
		{
			if ( p == -1 ) 
			{
				FillBuffer();
			}
			StringBuilder buf = new StringBuilder();
			for (int i = 0; i < nodes.Count; i++) 
			{
				object t = (object) nodes[i];
				buf.Append(" ");
				buf.Append(adaptor.GetNodeType(t));
			}
			return buf.ToString();
		}

		/** Debugging */
		public String ToTokenString(int start, int stop) {
			if ( p==-1 ) {
				FillBuffer();
			}
			StringBuilder buf = new StringBuilder();
			for (int i = start; i < nodes.Count && i <= stop; i++) {
				Object t = (Object) nodes[i];
				buf.Append(" ");
				buf.Append(adaptor.GetToken(t));
			}
			return buf.ToString();
		}

		public virtual string ToString(object start, object stop)
		{
			Console.Out.WriteLine("ToString");
			if ( (start == null) || (stop == null) ) 
			{
				return null;
			}
			if ( p == -1 ) 
			{
				FillBuffer();
			}
			//Console.Out.WriteLine("stop: " + stop);
			if ( start is CommonTree )
				Console.Out.Write("ToString: " + ((CommonTree)start).Token + ", ");
			else
				Console.Out.WriteLine(start);
			if ( stop is CommonTree )
				Console.Out.WriteLine(((CommonTree)stop).Token);
			else
				Console.Out.WriteLine(stop);
			// if we have the token stream, use that to dump text in order
			if ( tokens != null ) 
			{
				int beginTokenIndex = adaptor.GetTokenStartIndex(start);
				int endTokenIndex = adaptor.GetTokenStopIndex(stop);
				// if it's a tree, use start/stop index from start node
				// else use token range from start/stop nodes
				if ( adaptor.GetNodeType(stop) == Token.UP ) 
				{
					endTokenIndex = adaptor.GetTokenStopIndex(start);
				}
				else if ( adaptor.GetNodeType(stop) == Token.EOF ) 
				{
					endTokenIndex = Count-2; // don't use EOF
				}
				return tokens.ToString(beginTokenIndex, endTokenIndex);
			}
			// walk nodes looking for start
			object t = null;
			int i = 0;
			for (; i < nodes.Count; i++) 
			{
				t = nodes[i];
				if ( t == start ) 
				{
					break;
				}
			}
			// now walk until we see stop, filling string buffer with text
			StringBuilder buf = new StringBuilder();
			t = nodes[i];
			string text;
			while ( t != stop ) 
			{
				text = adaptor.GetNodeText(t);
				if ( text == null ) 
				{
					text = " " + adaptor.GetNodeType(t);
				}
				buf.Append(text);
				i++;
				t = nodes[i];
			}
			// include stop node too
			text = adaptor.GetNodeText(stop);
			if ( text == null ) 
			{
				text = " " + adaptor.GetNodeType(stop);
			}
			buf.Append(text);
			return buf.ToString();
		}
		
	}
}