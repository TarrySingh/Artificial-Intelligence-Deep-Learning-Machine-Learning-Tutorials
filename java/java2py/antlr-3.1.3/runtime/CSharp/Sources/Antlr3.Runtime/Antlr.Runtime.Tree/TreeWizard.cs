/*
[The "BSD licence"]
Copyright (c) 2005-2007 Kunle Odutola
Copyright (c) 2007 Johannes Luber
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
	using IList = System.Collections.IList;
	using IDictionary = System.Collections.IDictionary;
	using Hashtable = System.Collections.Hashtable;
	using ArrayList = System.Collections.ArrayList;
	using IToken = Antlr.Runtime.IToken;
	using Token = Antlr.Runtime.Token;

	/// <summary>
	/// Build and navigate trees with this object.  Must know about the names
	/// of tokens so you have to pass in a map or array of token names (from which
	/// this class can build the map).  I.e., Token DECL means nothing unless the
	/// class can translate it to a token type.
	/// </summary>
	/// <remarks>
	/// In order to create nodes and navigate, this class needs a TreeAdaptor.
	///
	/// This class can build a token type -> node index for repeated use or for
	/// iterating over the various nodes with a particular type.
	///
	/// This class works in conjunction with the TreeAdaptor rather than moving
	/// all this functionality into the adaptor.  An adaptor helps build and
	/// navigate trees using methods.  This class helps you do it with string
	/// patterns like "(A B C)".  You can create a tree from that pattern or
	/// match subtrees against it.
	/// </remarks>
	public class TreeWizard
	{
		protected ITreeAdaptor adaptor;
		protected IDictionary tokenNameToTypeMap;

		public interface ContextVisitor
		{
			void Visit(object t, object parent, int childIndex, IDictionary labels);
		}

		public abstract class Visitor : ContextVisitor
		{
			public void Visit(object t, object parent, int childIndex, IDictionary labels)
			{
				Visit(t);
			}
			public abstract void Visit(object t);
		}

		private sealed class RecordAllElementsVisitor : Visitor
		{
			private IList list;

			public RecordAllElementsVisitor(IList list)
			{
				this.list = list;
			}

			override public void Visit(object t)
			{
				list.Add(t);
			}
		}

		private sealed class PatternMatchingContextVisitor : ContextVisitor
		{
			private TreeWizard owner;
			private TreePattern pattern;
			private IList list;

			public PatternMatchingContextVisitor(TreeWizard owner, TreePattern pattern, IList list)
			{
				this.owner = owner;
				this.pattern = pattern;
				this.list = list;
			}

			public void Visit(object t, object parent, int childIndex, IDictionary labels)
			{
				if (owner._Parse(t, pattern, null))
				{
					list.Add(t);
				}
			}
		}

		private sealed class InvokeVisitorOnPatternMatchContextVisitor : ContextVisitor
		{
			private TreeWizard owner;
			private TreePattern pattern;
			private ContextVisitor visitor;
			private Hashtable labels = new Hashtable();

			public InvokeVisitorOnPatternMatchContextVisitor(TreeWizard owner, TreePattern pattern, ContextVisitor visitor)
			{
				this.owner = owner;
				this.pattern = pattern;
				this.visitor = visitor;
			}

			public void Visit(object t, object parent, int childIndex, IDictionary unusedlabels)
			{
				// the unusedlabels arg is null as visit on token type doesn't set.
				labels.Clear();
				if (owner._Parse(t, pattern, labels))
				{
					visitor.Visit(t, parent, childIndex, labels);
				}
			}
		}

		/// <summary>
		/// When using %label:TOKENNAME in a tree for parse(), we must track the label.
		/// </summary>
		public class TreePattern : CommonTree
		{
			public string label;
			public bool hasTextArg;
			public TreePattern(IToken payload)
				: base(payload)
			{
			}

			override public string ToString()
			{
				if (label != null)
				{
					return "%" + label + ":" + base.ToString();
				}
				else
				{
					return base.ToString();
				}
			}
		}

		public class WildcardTreePattern : TreePattern
		{
			public WildcardTreePattern(IToken payload)
				: base(payload)
			{
			}
		}

		/// <summary>
		/// This adaptor creates TreePattern objects for use during scan()
		/// </summary>
		public class TreePatternTreeAdaptor : CommonTreeAdaptor
		{
			override public object Create(IToken payload)
			{
				return new TreePattern(payload);
			}
		}

		public TreeWizard(ITreeAdaptor adaptor)
		{
			this.adaptor = adaptor;
		}

		public TreeWizard(ITreeAdaptor adaptor, IDictionary tokenNameToTypeMap)
		{
			this.adaptor = adaptor;
			this.tokenNameToTypeMap = tokenNameToTypeMap;
		}

		public TreeWizard(ITreeAdaptor adaptor, string[] tokenNames)
		{
			this.adaptor = adaptor;
			this.tokenNameToTypeMap = ComputeTokenTypes(tokenNames);
		}

		public TreeWizard(string[] tokenNames)
			: this(null, tokenNames)
		{
		}

		/// <summary>
		/// Compute a Map&lt;String, Integer&gt; that is an inverted index of
		/// tokenNames (which maps int token types to names).
		/// </summary>
		public IDictionary ComputeTokenTypes(string[] tokenNames)
		{
			IDictionary m = new Hashtable();
			if (tokenNames == null) {
				return m;
			}
			for (int ttype = Token.MIN_TOKEN_TYPE; ttype < tokenNames.Length; ttype++)
			{
				string name = tokenNames[ttype];
				m.Add(name, ttype);
			}
			return m;
		}

		/// <summary>
		/// Using the map of token names to token types, return the type.
		/// </summary>
		public int GetTokenType(string tokenName)
		{
			if (tokenNameToTypeMap == null)
			{
				return Token.INVALID_TOKEN_TYPE;
			}
			object ttypeI = tokenNameToTypeMap[tokenName];
			if (ttypeI != null)
			{
				return (int)ttypeI;
			}
			return Token.INVALID_TOKEN_TYPE;
		}

		/// <summary>
		/// Walk the entire tree and make a node name to nodes mapping.
		/// </summary>
		/// <remarks>
		/// For now, use recursion but later nonrecursive version may be
		/// more efficient.  Returns Map&lt;Integer, List&gt; where the List is
		/// of your AST node type.  The Integer is the token type of the node.
		///
		/// TODO: save this index so that find and visit are faster
		/// </remarks>
		public IDictionary Index(object t)
		{
			IDictionary m = new Hashtable();
			_Index(t, m);
			return m;
		}

		/// <summary>Do the work for index</summary>
		protected void _Index(object t, IDictionary m)
		{
			if (t == null)
			{
				return;
			}
			int ttype = adaptor.GetNodeType(t);
			IList elements = m[ttype] as IList;
			if (elements == null)
			{
				elements = new ArrayList();
				m[ttype] = elements;
			}
			elements.Add(t);
			int n = adaptor.GetChildCount(t);
			for (int i = 0; i < n; i++)
			{
				object child = adaptor.GetChild(t, i);
				_Index(child, m);
			}
		}

		/// <summary>Return a List of tree nodes with token type ttype</summary>
		public IList Find(object t, int ttype)
		{
			IList nodes = new ArrayList();
			Visit(t, ttype, new TreeWizard.RecordAllElementsVisitor(nodes));
			return nodes;
		}

		/// <summary>Return a List of subtrees matching pattern</summary>
		public IList Find(object t, string pattern)
		{
			IList subtrees = new ArrayList();
			// Create a TreePattern from the pattern
			TreePatternLexer tokenizer = new TreePatternLexer(pattern);
			TreePatternParser parser = new TreePatternParser(tokenizer, this, new TreePatternTreeAdaptor());
			TreePattern tpattern = (TreePattern)parser.Pattern();
			// don't allow invalid patterns
			if ((tpattern == null) || tpattern.IsNil || (tpattern.GetType() == typeof(WildcardTreePattern)))
			{
				return null;
			}
			int rootTokenType = tpattern.Type;
			Visit(t, rootTokenType, new TreeWizard.PatternMatchingContextVisitor(this, tpattern, subtrees));
			return subtrees;
		}

		public object FindFirst(object t, int ttype)
		{
			return null;
		}

		public object FindFirst(object t, string pattern)
		{
			return null;
		}

		/// <summary>
		/// Visit every ttype node in t, invoking the visitor.
		/// </summary>
		/// <remarks>
		/// This is a quicker
		/// version of the general visit(t, pattern) method.  The labels arg
		/// of the visitor action method is never set (it's null) since using
		/// a token type rather than a pattern doesn't let us set a label.
		/// </remarks>
		public void Visit(object t, int ttype, ContextVisitor visitor)
		{
			_Visit(t, null, 0, ttype, visitor);
		}

		/// <summary>Do the recursive work for visit</summary>
		protected void _Visit(object t, object parent, int childIndex, int ttype, ContextVisitor visitor)
		{
			if (t == null)
			{
				return;
			}
			if (adaptor.GetNodeType(t) == ttype)
			{
				visitor.Visit(t, parent, childIndex, null);
			}
			int n = adaptor.GetChildCount(t);
			for (int i = 0; i < n; i++)
			{
				object child = adaptor.GetChild(t, i);
				_Visit(child, t, i, ttype, visitor);
			}
		}

		/// <summary>
		/// For all subtrees that match the pattern, execute the visit action.
		/// </summary>
		/// <remarks>
		/// The implementation uses the root node of the pattern in combination
		/// with visit(t, ttype, visitor) so nil-rooted patterns are not allowed.
		/// Patterns with wildcard roots are also not allowed.
		/// </remarks>
		public void Visit(object t, string pattern, ContextVisitor visitor)
		{
			// Create a TreePattern from the pattern
			TreePatternLexer tokenizer = new TreePatternLexer(pattern);
			TreePatternParser parser = new TreePatternParser(tokenizer, this, new TreePatternTreeAdaptor());
			TreePattern tpattern = (TreePattern)parser.Pattern();
			// don't allow invalid patterns
			if ((tpattern == null) || tpattern.IsNil || (tpattern.GetType() == typeof(WildcardTreePattern)))
			{
				return;
			}
			//IDictionary labels = new Hashtable(); // reused for each _parse
			int rootTokenType = tpattern.Type;
			Visit(t, rootTokenType,
				new TreeWizard.InvokeVisitorOnPatternMatchContextVisitor(this, tpattern, visitor));
		}

		/// <summary>
		/// Given a pattern like (ASSIGN %lhs:ID %rhs:.) with optional labels
		/// on the various nodes and '.' (dot) as the node/subtree wildcard,
		/// return true if the pattern matches and fill the labels Map with
		/// the labels pointing at the appropriate nodes.  Return false if
		/// the pattern is malformed or the tree does not match.
		/// </summary>
		/// <remarks>
		/// If a node specifies a text arg in pattern, then that must match
		/// for that node in t.
		/// 
		/// TODO: what's a better way to indicate bad pattern? Exceptions are a hassle 
		/// </remarks>
		public bool Parse(object t, string pattern, IDictionary labels)
		{
			TreePatternLexer tokenizer = new TreePatternLexer(pattern);
			TreePatternParser parser = new TreePatternParser(tokenizer, this, new TreePatternTreeAdaptor());
			TreePattern tpattern = (TreePattern)parser.Pattern();

			bool matched = _Parse(t, tpattern, labels);
			return matched;
		}

		public bool Parse(object t, string pattern)
		{
			return Parse(t, pattern, null);
		}

		/// <summary>
		/// Do the work for Parse(). Check to see if the t2 pattern fits the
		/// structure and token types in t1.  Check text if the pattern has
		/// text arguments on nodes.  Fill labels map with pointers to nodes
		/// in tree matched against nodes in pattern with labels.
		/// </summary>
		protected bool _Parse(object t1, TreePattern t2, IDictionary labels)
		{
			// make sure both are non-null
			if (t1 == null || t2 == null)
			{
				return false;
			}
			// check roots (wildcard matches anything)
			if (t2.GetType() != typeof(WildcardTreePattern))
			{
				if (adaptor.GetNodeType(t1) != t2.Type)
				{
					return false;
				}
				if (t2.hasTextArg && !adaptor.GetNodeText(t1).Equals(t2.Text))
				{
					return false;
				}
			}
			if (t2.label != null && labels != null)
			{
				// map label in pattern to node in t1
				labels[t2.label] = t1;
			}
			// check children
			int n1 = adaptor.GetChildCount(t1);
			int n2 = t2.ChildCount;
			if (n1 != n2)
			{
				return false;
			}
			for (int i = 0; i < n1; i++)
			{
				object child1 = adaptor.GetChild(t1, i);
				TreePattern child2 = (TreePattern)t2.GetChild(i);
				if (!_Parse(child1, child2, labels))
				{
					return false;
				}
			}
			return true;
		}

		/// <summary>
		/// Create a tree or node from the indicated tree pattern that closely
		/// follows ANTLR tree grammar tree element syntax:
		///
		///		(root child1 ... child2).
		///
		/// </summary>
		/// <remarks>
		/// You can also just pass in a node: ID
		///
		/// Any node can have a text argument: ID[foo]
		/// (notice there are no quotes around foo--it's clear it's a string).
		///
		/// nil is a special name meaning "give me a nil node".  Useful for
		/// making lists: (nil A B C) is a list of A B C.
		/// </remarks>
		public object Create(string pattern)
		{
			TreePatternLexer tokenizer = new TreePatternLexer(pattern);
			TreePatternParser parser = new TreePatternParser(tokenizer, this, adaptor);
			object t = parser.Pattern();
			return t;
		}

		/// <summary>
		/// Compare t1 and t2; return true if token types/text, structure match exactly.
		/// The trees are examined in their entirety so that (A B) does not match
		/// (A B C) nor (A (B C)). 
		/// </summary>
		/// <remarks>
		/// TODO: allow them to pass in a comparator
		/// TODO: have a version that is nonstatic so it can use instance adaptor
		/// 
		/// I cannot rely on the tree node's equals() implementation as I make
		/// no constraints at all on the node types nor interface etc... 
		/// </remarks>
		public static bool Equals(object t1, object t2, ITreeAdaptor adaptor)
		{
			return _Equals(t1, t2, adaptor);
		}

		/// <summary>
		/// Compare type, structure, and text of two trees, assuming adaptor in
		/// this instance of a TreeWizard.
		/// </summary>
		public new bool Equals(object t1, object t2)
		{
			return _Equals(t1, t2, adaptor);
		}

		protected static bool _Equals(object t1, object t2, ITreeAdaptor adaptor)
		{
			// make sure both are non-null
			if (t1 == null || t2 == null)
			{
				return false;
			}
			// check roots
			if (adaptor.GetNodeType(t1) != adaptor.GetNodeType(t2))
			{
				return false;
			}
			if (!adaptor.GetNodeText(t1).Equals(adaptor.GetNodeText(t2)))
			{
				return false;
			}
			// check children
			int n1 = adaptor.GetChildCount(t1);
			int n2 = adaptor.GetChildCount(t2);
			if (n1 != n2)
			{
				return false;
			}
			for (int i = 0; i < n1; i++)
			{
				object child1 = adaptor.GetChild(t1, i);
				object child2 = adaptor.GetChild(t2, i);
				if (!_Equals(child1, child2, adaptor))
				{
					return false;
				}
			}
			return true;
		}
	}
}