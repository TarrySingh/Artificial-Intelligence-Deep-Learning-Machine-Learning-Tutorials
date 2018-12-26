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


namespace Antlr.Utility.Tree
{
	using System;
	using IDictionary		= System.Collections.IDictionary;
	using Hashtable			= System.Collections.Hashtable;
	using StringTemplate	= Antlr.StringTemplate.StringTemplate;
	using ITree				= Antlr.Runtime.Tree.ITree;
	using ITreeAdaptor		= Antlr.Runtime.Tree.ITreeAdaptor;
	using CommonTreeAdaptor	= Antlr.Runtime.Tree.CommonTreeAdaptor;
	
	/// <summary>
	/// A utility class to generate DOT diagrams (graphviz) from
	/// arbitrary trees.  You can pass in your own templates and
	/// can pass in any kind of tree or use Tree interface method.
	/// I wanted this separator so that you don't have to include
	/// ST just to use the org.antlr.runtime.tree.* package.
	/// This is a set of non-static methods so you can subclass
	/// to override.  For example, here is an invocation:
	///
	///     CharStream input = new ANTLRInputStream(Console.In);
	///     TLexer lex = new TLexer(input);
	///     CommonTokenStream tokens = new CommonTokenStream(lex);
	///     TParser parser = new TParser(tokens);
	///     TParser.e_return r = parser.e();
	///     Tree t = (Tree)r.tree;
	///     Console.Out.WriteLine(t.ToStringTree());
	///     DOTTreeGenerator gen = new DOTTreeGenerator();
	///     StringTemplate st = gen.ToDOT(t);
	///     Console.Out.WriteLine(st);
	///     
	/// </summary>
	public class DOTTreeGenerator
	{
		public static StringTemplate _treeST = new StringTemplate(
			"digraph {\n" +
			"  ordering=out;\n" +
			"  ranksep=.4;\n" +
			"  node [shape=plaintext, fixedsize=true, fontsize=11, fontname=\"Courier\",\n" +
			"        width=.25, height=.25];\n" +
			"  edge [arrowsize=.5]\n" +
			"  $nodes$\n" +
			"  $edges$\n" +
			"}\n"
			);

		public static StringTemplate _nodeST = new StringTemplate("$name$ [label=\"$text$\"];\n");

		public static StringTemplate _edgeST = new StringTemplate("$parent$ -> $child$ // \"$parentText$\" -> \"$childText$\"\n");

		/// <summary>
		/// Track node to number mapping so we can get proper node name back
		/// </summary>
		IDictionary nodeToNumberMap = new Hashtable();

		/// <summary>
		/// Track node number so we can get unique node names
		/// </summary>
		int nodeNumber = 0;

		public StringTemplate ToDOT(object tree, ITreeAdaptor adaptor, StringTemplate _treeST, StringTemplate _edgeST)
		{
			StringTemplate treeST = _treeST.GetInstanceOf();
			nodeNumber = 0;
			ToDOTDefineNodes(tree, adaptor, treeST);
			nodeNumber = 0;
			ToDOTDefineEdges(tree, adaptor, treeST);
			/*
			if ( adaptor.GetChildCount(tree)==0 ) {
				// single node, don't do edge.
				treeST.SetAttribute("nodes", adaptor.GetNodeText(tree));
			}
			*/
			return treeST;
		}

		public StringTemplate ToDOT(object tree, ITreeAdaptor adaptor)
		{
			return ToDOT(tree, adaptor, _treeST, _edgeST);
		}

		/// <summary>
		/// Generate DOT (graphviz) for a whole tree not just a node.
		/// For example, 3+4*5 should generate:
		///
		/// digraph {
		///   node [shape=plaintext, fixedsize=true, fontsize=11, fontname="Courier",
		///         width=.4, height=.2];
		///   edge [arrowsize=.7]
		///   "+"->3
		///   "+"->"*"
		///   "*"->4
		///   "*"->5
		/// }
		///
		/// Return the ST not a string in case people want to alter.
		///
		/// Takes a Tree interface object.
		/// </summary>
		public StringTemplate ToDOT(ITree tree) 
		{
			return ToDOT(tree, new CommonTreeAdaptor());
		}

		protected void ToDOTDefineNodes(object tree, ITreeAdaptor adaptor, StringTemplate treeST)
		{
			if ( tree == null ) 
			{
				return;
			}
			int n = adaptor.GetChildCount(tree);
			if ( n == 0 ) 
			{
				// must have already dumped as child from previous
				// invocation; do nothing
				return;
			}

			// define parent node
			StringTemplate parentNodeST = GetNodeST(adaptor, tree);
			treeST.SetAttribute("nodes", parentNodeST);

			// for each child, do a "<unique-name> [label=text]" node def
			for (int i = 0; i < n; i++) 
			{
				object child = adaptor.GetChild(tree, i);
				StringTemplate nodeST = GetNodeST(adaptor, child);
				treeST.SetAttribute("nodes", nodeST);
				ToDOTDefineNodes(child, adaptor, treeST);
			}
		}

		protected void ToDOTDefineEdges(object tree, ITreeAdaptor adaptor, StringTemplate treeST)
		{
			if ( tree == null ) 
			{
				return;
			}
			int n = adaptor.GetChildCount(tree);
			if ( n == 0 ) 
			{
				// must have already dumped as child from previous
				// invocation; do nothing
				return;
			}

			string parentName = "n" + GetNodeNumber(tree);

			// for each child, do a parent -> child edge using unique node names
			string parentText = adaptor.GetNodeText(tree);
			for (int i = 0; i < n; i++) 
			{
				object child = adaptor.GetChild(tree, i);
				string childText = adaptor.GetNodeText(child);
				string childName = "n" + GetNodeNumber(child);
				StringTemplate edgeST = _edgeST.GetInstanceOf();
				edgeST.SetAttribute("parent", parentName);
				edgeST.SetAttribute("child", childName);
				edgeST.SetAttribute("parentText", parentText);
				edgeST.SetAttribute("childText", childText);
				treeST.SetAttribute("edges", edgeST);
				ToDOTDefineEdges(child, adaptor, treeST);
			}
		}

		protected StringTemplate GetNodeST(ITreeAdaptor adaptor, object t) 
		{
			string text = adaptor.GetNodeText(t);
			StringTemplate nodeST = _nodeST.GetInstanceOf();
			string uniqueName = "n" + GetNodeNumber(t);
			nodeST.SetAttribute("name", uniqueName);
			if (text != null) 
				text = text.Replace("\"", "\\\\\"");
			nodeST.SetAttribute("text", text);
			return nodeST;
		}

		protected int GetNodeNumber(object t) 
		{
			object boxedInt = nodeToNumberMap[t];
			if ( boxedInt != null ) 
			{
				return (int)boxedInt;
			}
			else 
			{
				nodeToNumberMap[t] = nodeNumber;
				nodeNumber++;
				return nodeNumber-1;
			}
		}
	}
}
