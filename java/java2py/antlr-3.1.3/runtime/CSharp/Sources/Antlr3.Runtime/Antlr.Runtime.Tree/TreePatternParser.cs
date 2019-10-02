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
	using IToken = Antlr.Runtime.IToken;
	using CommonToken = Antlr.Runtime.CommonToken;

	public class TreePatternParser
	{
		protected TreePatternLexer tokenizer;
		protected int ttype;
		protected TreeWizard wizard;
		protected ITreeAdaptor adaptor;

		public TreePatternParser(TreePatternLexer tokenizer, TreeWizard wizard, ITreeAdaptor adaptor)
		{
			this.tokenizer = tokenizer;
			this.wizard = wizard;
			this.adaptor = adaptor;
			ttype = tokenizer.NextToken(); // kickstart
		}

		public object Pattern()
		{
			if (ttype == TreePatternLexer.BEGIN)
			{
				return ParseTree();
			}
			else if (ttype == TreePatternLexer.ID)
			{
				object node = ParseNode();
				if (ttype == TreePatternLexer.EOF)
				{
					return node;
				}
				return null; // extra junk on end
			}
			return null;
		}

		public object ParseTree()
		{
			if (ttype != TreePatternLexer.BEGIN)
			{
				Console.Out.WriteLine("no BEGIN");
				return null;
			}
			ttype = tokenizer.NextToken();
			object root = ParseNode();
			if (root == null)
			{
				return null;
			}
			while (ttype == TreePatternLexer.BEGIN ||
					ttype == TreePatternLexer.ID ||
					ttype == TreePatternLexer.PERCENT ||
					ttype == TreePatternLexer.DOT)
			{
				if (ttype == TreePatternLexer.BEGIN)
				{
					object subtree = ParseTree();
					adaptor.AddChild(root, subtree);
				}
				else
				{
					object child = ParseNode();
					if (child == null)
					{
						return null;
					}
					adaptor.AddChild(root, child);
				}
			}
			if (ttype != TreePatternLexer.END)
			{
				Console.Out.WriteLine("no END");
				return null;
			}
			ttype = tokenizer.NextToken();
			return root;
		}

		public object ParseNode()
		{
			// "%label:" prefix
			string label = null;
			if (ttype == TreePatternLexer.PERCENT)
			{
				ttype = tokenizer.NextToken();
				if (ttype != TreePatternLexer.ID)
				{
					return null;
				}
				label = tokenizer.sval.ToString();
				ttype = tokenizer.NextToken();
				if (ttype != TreePatternLexer.COLON)
				{
					return null;
				}
				ttype = tokenizer.NextToken(); // move to ID following colon
			}

			// Wildcard?
			if (ttype == TreePatternLexer.DOT)
			{
				ttype = tokenizer.NextToken();
				IToken wildcardPayload = new CommonToken(0, ".");
				TreeWizard.TreePattern node = new TreeWizard.WildcardTreePattern(wildcardPayload);
				if (label != null)
				{
					node.label = label;
				}
				return node;
			}

			// "ID" or "ID[arg]"
			if (ttype != TreePatternLexer.ID)
			{
				return null;
			}
			string tokenName = tokenizer.sval.ToString();
			ttype = tokenizer.NextToken();
			if (tokenName.Equals("nil"))
			{
				return adaptor.GetNilNode();
			}
			string text = tokenName;
			// check for arg
			string arg = null;
			if (ttype == TreePatternLexer.ARG)
			{
				arg = tokenizer.sval.ToString();
				text = arg;
				ttype = tokenizer.NextToken();
			}

			// create node
			int treeNodeType = wizard.GetTokenType(tokenName);
			if (treeNodeType == Token.INVALID_TOKEN_TYPE)
			{
				return null;
			}
			object nodeObj;
			nodeObj = adaptor.Create(treeNodeType, text);
			if ((label != null) && (nodeObj.GetType() == typeof(TreeWizard.TreePattern)))
			{
				((TreeWizard.TreePattern)nodeObj).label = label;
			}
			if ((arg != null) && (nodeObj.GetType() == typeof(TreeWizard.TreePattern)))
			{
				((TreeWizard.TreePattern)nodeObj).hasTextArg = true;
			}
			return nodeObj;
		}
	}
}