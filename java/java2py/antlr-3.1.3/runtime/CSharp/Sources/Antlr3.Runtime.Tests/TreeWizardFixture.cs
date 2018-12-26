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


namespace Antlr.Runtime.Tests
{
	using System;
	using IList = System.Collections.IList;
	using IDictionary = System.Collections.IDictionary;
	using ArrayList = System.Collections.ArrayList;
	using Hashtable = System.Collections.Hashtable;
	using StringBuilder = System.Text.StringBuilder;

	using IToken = Antlr.Runtime.IToken;
	using Token = Antlr.Runtime.Token;
	using CommonToken = Antlr.Runtime.CommonToken;
	using ITree = Antlr.Runtime.Tree.ITree;
	using TreeWizard = Antlr.Runtime.Tree.TreeWizard;
	using CommonTree = Antlr.Runtime.Tree.CommonTree;
	using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;
	using CommonTreeAdaptor = Antlr.Runtime.Tree.CommonTreeAdaptor;
	using CollectionUtils = Antlr.Runtime.Collections.CollectionUtils;

	using MbUnit.Framework;

	[TestFixture]
	public class TreeWizardFixture : TestFixtureBase
	{
		protected static readonly String[] tokens =
			new String[] { "", "", "", "", "", "A", "B", "C", "D", "E", "ID", "VAR" };

		private sealed class RecordAllElementsVisitor : TreeWizard.Visitor
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

		private sealed class Test1ContextVisitor : TreeWizard.ContextVisitor
		{
			private ITreeAdaptor adaptor;
			private IList list;

			public Test1ContextVisitor(ITreeAdaptor adaptor, IList list)
			{
				this.adaptor = adaptor;
				this.list = list;
			}

			public void Visit(object t, object parent, int childIndex, IDictionary labels)
			{
				list.Add(adaptor.GetNodeText(t)
					+ "@" + ((parent != null) ? adaptor.GetNodeText(parent) : "nil")
					+ "[" + childIndex + "]");
			}
		}

		private sealed class Test2ContextVisitor : TreeWizard.ContextVisitor
		{
			private ITreeAdaptor adaptor;
			private IList list;

			public Test2ContextVisitor(ITreeAdaptor adaptor, IList list)
			{
				this.adaptor = adaptor;
				this.list = list;
			}

			public void Visit(object t, object parent, int childIndex, IDictionary labels)
			{

				list.Add(adaptor.GetNodeText(t)
					+ "@" + ((parent != null) ? adaptor.GetNodeText(parent) : "nil")
					+ "[" + childIndex + "]" + labels["a"] + "&" + labels["b"]);
			}
		}

		#region TreeWizard Tests

		[Test]
		public void testSingleNode()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("ID");
			string actual = t.ToStringTree();
			string expected = "ID";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testSingleNodeWithArg()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("ID[foo]");
			string actual = t.ToStringTree();
			string expected = "foo";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testSingleNodeTree()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A)");
			string actual = t.ToStringTree();
			string expected = "A";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testSingleLevelTree()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C D)");
			string actual = t.ToStringTree();
			string expected = "(A B C D)";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testListTree()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(nil A B C)");
			string actual = t.ToStringTree();
			string expected = "A B C";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testInvalidListTree()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("A B C");
			Assert.IsTrue(t == null);
		}

		[Test]
		public void testDoubleLevelTree()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A (B C) (B D) E)");
			string actual = t.ToStringTree();
			string expected = "(A (B C) (B D) E)";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testSingleNodeIndex()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("ID");
			IDictionary m = wiz.Index(t);
			string actual = CollectionUtils.DictionaryToString(m);
			string expected = "{10=[ID]}";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testNoRepeatsIndex()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C D)");
			IDictionary m = wiz.Index(t);
			string actual = CollectionUtils.DictionaryToString(m);
			//string expected = "{8=[D], 6=[B], 7=[C], 5=[A]}";
			string expected = "{8=[D], 7=[C], 6=[B], 5=[A]}";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testRepeatsIndex()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B (A C B) B D D)");
			IDictionary m = wiz.Index(t);
			string actual = CollectionUtils.DictionaryToString(m);
			//string expected = "{8=[D, D], 6=[B, B, B], 7=[C], 5=[A, A]}";
			string expected = "{8=[D, D], 7=[C], 6=[B, B, B], 5=[A, A]}";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testNoRepeatsVisit()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C D)");
			IList elements = new ArrayList();
			wiz.Visit(t, wiz.GetTokenType("B"), new RecordAllElementsVisitor(elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[B]";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testNoRepeatsVisit2()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B (A C B) B D D)");
			IList elements = new ArrayList();
			wiz.Visit(t, wiz.GetTokenType("C"), new RecordAllElementsVisitor(elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[C]";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testRepeatsVisit()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B (A C B) B D D)");
			IList elements = new ArrayList();
			wiz.Visit(t, wiz.GetTokenType("B"), new RecordAllElementsVisitor(elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[B, B, B]";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testRepeatsVisit2()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B (A C B) B D D)");
			IList elements = new ArrayList();
			wiz.Visit(t, wiz.GetTokenType("A"), new RecordAllElementsVisitor(elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[A, A]";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testRepeatsVisitWithContext()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B (A C B) B D D)");
			IList elements = new ArrayList();
			wiz.Visit(t, wiz.GetTokenType("B"), new Test1ContextVisitor(adaptor, elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[B@A[0], B@A[1], B@A[2]]";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testRepeatsVisitWithNullParentAndContext()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B (A C B) B D D)");
			IList elements = new ArrayList();
			wiz.Visit(t, wiz.GetTokenType("A"), new Test1ContextVisitor(adaptor, elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[A@nil[0], A@A[1]]";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testVisitPattern()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C (A B) D)");
			IList elements = new ArrayList();
			wiz.Visit(t, "(A B)", new RecordAllElementsVisitor(elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[A]"; // shouldn't match overall root, just (A B)
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testVisitPatternMultiple()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C (A B) (D (A B)))");
			IList elements = new ArrayList();
			wiz.Visit(t, "(A B)", new Test1ContextVisitor(adaptor, elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[A@A[2], A@D[0]]"; // shouldn't match overall root, just (A B)
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testVisitPatternMultipleWithLabels()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C (A[foo] B[bar]) (D (A[big] B[dog])))");
			IList elements = new ArrayList();
			wiz.Visit(t, "(%a:A %b:B)", new Test2ContextVisitor(adaptor, elements));
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[foo@A[2]foo&bar, big@D[0]big&dog]";
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testParse()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C)");
			bool valid = wiz.Parse(t, "(A B C)");
			Assert.IsTrue(valid);
		}

		[Test]
		public void testParseSingleNode()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("A");
			bool valid = wiz.Parse(t, "A");
			Assert.IsTrue(valid);
		}

		[Test]
		public void testParseFlatTree()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(nil A B C)");
			bool valid = wiz.Parse(t, "(nil A B C)");
			Assert.IsTrue(valid);
		}

		[Test]
		public void testWildcard()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C)");
			bool valid = wiz.Parse(t, "(A . .)");
			Assert.IsTrue(valid);
		}

		[Test]
		public void testParseWithText()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B[foo] C[bar])");
			// C pattern has no text arg so despite [bar] in t, no need
			// to match text--check structure only.
			bool valid = wiz.Parse(t, "(A B[foo] C)");
			Assert.IsTrue(valid);
		}

		[Test]
		public void testParseWithTextFails()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C)");
			bool valid = wiz.Parse(t, "(A[foo] B C)");
			Assert.IsTrue(!valid); // fails
		}

		[Test]
		public void testParseLabels()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C)");
			IDictionary labels = new Hashtable();
			bool valid = wiz.Parse(t, "(%a:A %b:B %c:C)", labels);
			Assert.IsTrue(valid);
			Assert.AreEqual("A", labels["a"].ToString());
			Assert.AreEqual("B", labels["b"].ToString());
			Assert.AreEqual("C", labels["c"].ToString());
		}

		[Test]
		public void testParseWithWildcardLabels()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C)");
			IDictionary labels = new Hashtable();
			bool valid = wiz.Parse(t, "(A %b:. %c:.)", labels);
			Assert.IsTrue(valid);
			Assert.AreEqual("B", labels["b"].ToString());
			Assert.AreEqual("C", labels["c"].ToString());
		}

		[Test]
		public void testParseLabelsAndTestText()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B[foo] C)");
			IDictionary labels = new Hashtable();
			bool valid = wiz.Parse(t, "(%a:A %b:B[foo] %c:C)", labels);
			Assert.IsTrue(valid);
			Assert.AreEqual("A", labels["a"].ToString());
			Assert.AreEqual("foo", labels["b"].ToString());
			Assert.AreEqual("C", labels["c"].ToString());
		}

		[Test]
		public void testParseLabelsInNestedTree()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A (B C) (D E))");
			IDictionary labels = new Hashtable();
			bool valid = wiz.Parse(t, "(%a:A (%b:B %c:C) (%d:D %e:E) )", labels);
			Assert.IsTrue(valid);
			Assert.AreEqual("A", labels["a"].ToString());
			Assert.AreEqual("B", labels["b"].ToString());
			Assert.AreEqual("C", labels["c"].ToString());
			Assert.AreEqual("D", labels["d"].ToString());
			Assert.AreEqual("E", labels["e"].ToString());
		}

		[Test]
		public void testEquals()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t1 = (CommonTree)wiz.Create("(A B C)");
			CommonTree t2 = (CommonTree)wiz.Create("(A B C)");
			bool same = TreeWizard.Equals(t1, t2, adaptor);
			Assert.IsTrue(same);
		}

		[Test]
		public void testEqualsWithText()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t1 = (CommonTree)wiz.Create("(A B[foo] C)");
			CommonTree t2 = (CommonTree)wiz.Create("(A B[foo] C)");
			bool same = TreeWizard.Equals(t1, t2, adaptor);
			Assert.IsTrue(same);
		}

		[Test]
		public void testEqualsWithMismatchedText()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t1 = (CommonTree)wiz.Create("(A B[foo] C)");
			CommonTree t2 = (CommonTree)wiz.Create("(A B C)");
			bool same = TreeWizard.Equals(t1, t2, adaptor);
			Assert.IsTrue(!same);
		}

		[Test]
		public void testFindPattern()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			TreeWizard wiz = new TreeWizard(adaptor, tokens);
			CommonTree t = (CommonTree)wiz.Create("(A B C (A[foo] B[bar]) (D (A[big] B[dog])))");
			IList subtrees = wiz.Find(t, "(A B)");
			IList elements = subtrees;
			string actual = CollectionUtils.ListToString(elements);
			string expected = "[foo, big]";
			Assert.AreEqual(expected, actual);
		}

		#endregion
	}
}