/*
[The "BSD licence"]
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

#pragma warning disable 219 // No unused variable warnings

namespace Antlr.Runtime.Tests {
	using System;
	using System.Collections.Generic;
	using Antlr.Runtime.Tree;

	using MbUnit.Framework;

	[TestFixture]
	public class RewriteRuleXxxxStreamFixture : TestFixtureBase {
		#region Check Constructors

		[Test]
		public void CheckRewriteRuleTokenStreamConstructors() {
			RewriteRuleTokenStream tokenTest1 = new RewriteRuleTokenStream(CreateTreeAdaptor(),
				"RewriteRuleTokenStream test1");

			RewriteRuleTokenStream tokenTest2 = new RewriteRuleTokenStream(CreateTreeAdaptor(),
				"RewriteRuleTokenStream test2", CreateToken(1,
				"test token without any real context"));

			RewriteRuleTokenStream tokenTest3 = new RewriteRuleTokenStream(CreateTreeAdaptor(),
				"RewriteRuleTokenStream test3", CreateTokenList(4));
		}

		[Test]
		public void CheckRewriteRuleSubtreeStreamConstructors() {
			RewriteRuleSubtreeStream subtreeTest1 =
				new RewriteRuleSubtreeStream(CreateTreeAdaptor(),
				"RewriteRuleSubtreeStream test1");

			RewriteRuleSubtreeStream subtreeTest2 =
				new RewriteRuleSubtreeStream(CreateTreeAdaptor(),
				"RewriteRuleSubtreeStream test2", CreateToken(1,
				"test token without any real context"));

			RewriteRuleSubtreeStream subtreeTest3 =
				new RewriteRuleSubtreeStream(CreateTreeAdaptor(),
				"RewriteRuleSubtreeStream test3", CreateTokenList(4));
		}

		[Test]
		public void CheckRewriteRuleNodeStreamConstructors() {
			RewriteRuleNodeStream nodeTest1 = new RewriteRuleNodeStream(CreateTreeAdaptor(),
				"RewriteRuleNodeStream test1");

			RewriteRuleNodeStream nodeTest2 = new RewriteRuleNodeStream(CreateTreeAdaptor(),
				"RewriteRuleNodeStream test2", CreateToken(1,
				"test token without any real context"));

			RewriteRuleNodeStream nodeTest3 = new RewriteRuleNodeStream(CreateTreeAdaptor(),
				"RewriteRuleNodeStream test3", CreateTokenList(4));
		}
		#endregion

		#region Method Tests

		#region Empty Behaviour
		[Test]
		[ExpectedException(typeof(RewriteEmptyStreamException), "RewriteRuleTokenStream test")]
		public void CheckRRTokenStreamBehaviourWhileEmpty1() {
			string description = "RewriteRuleTokenStream test";
			RewriteRuleTokenStream tokenTest =
				new RewriteRuleTokenStream(CreateTreeAdaptor(),	description);

			Assert.IsFalse(tokenTest.HasNext(), "tokenTest has to give back false here.");
			Assert.AreEqual(description.ToString(), tokenTest.Description,
				"Description strings should be equal.");
			Assert.AreEqual(0, tokenTest.Size(), "The number of elements should be zero.");
			tokenTest.Reset();
			Assert.IsTrue(true, "Reset() shouldn't make any problems here.");
			Assert.AreEqual(0, tokenTest.Size(),
				"The number of elements should be still zero.");
			tokenTest.NextNode();
		}

		[Test]
		[ExpectedException(typeof(RewriteEmptyStreamException),
			"RewriteRuleSubtreeStream test")]
		public void CheckRRSubtreeStreamBehaviourWhileEmpty1() {
			string description = "RewriteRuleSubtreeStream test";
			RewriteRuleSubtreeStream subtreeTest =
				new RewriteRuleSubtreeStream(CreateTreeAdaptor(), description);

			Assert.IsFalse(subtreeTest.HasNext(), "HasNext() has to give back false here.");
			Assert.AreEqual(description.ToString(), subtreeTest.Description,
				"Description strings should be equal.");
			Assert.AreEqual(0, subtreeTest.Size(), "The number of elements should be zero.");
			subtreeTest.Reset();
			Assert.IsTrue(true, "Reset() shouldn't make any problems here.");
			Assert.AreEqual(0, subtreeTest.Size(),
				"The number of elements should be still zero.");
			subtreeTest.NextNode();
		}

		[Test]
		[ExpectedException(typeof(RewriteEmptyStreamException), "RewriteRuleNodeStream test")]
		public void CheckRRNodeStreamBehaviourWhileEmpty1() {
			string description = "RewriteRuleNodeStream test";
			RewriteRuleNodeStream nodeTest =
				new RewriteRuleNodeStream(CreateTreeAdaptor(), description);

			Assert.IsFalse(nodeTest.HasNext(), "HasNext() has to give back false here.");
			Assert.AreEqual(description.ToString(), nodeTest.Description,
				"Description strings should be equal.");
			Assert.AreEqual(0, nodeTest.Size(), "The number of elements should be zero.");
			nodeTest.Reset();
			Assert.IsTrue(true, "Reset() shouldn't make any problems here.");
			Assert.AreEqual(0, nodeTest.Size(),
				"The number of elements should be still zero.");
			nodeTest.NextNode();
		}

		[Test]
		[ExpectedException(typeof(RewriteEmptyStreamException), "RewriteRuleTokenStream test")]
		public void CheckRRTokenStreamBehaviourWhileEmpty2() {
			RewriteRuleTokenStream tokenTest = new RewriteRuleTokenStream(CreateTreeAdaptor(),
				"RewriteRuleTokenStream test");

			tokenTest.NextTree();
		}

		[Test]
		[ExpectedException(typeof(RewriteEmptyStreamException),
			"RewriteRuleSubtreeStream test")]
		public void CheckRRSubtreeStreamBehaviourWhileEmpty2() {
			RewriteRuleSubtreeStream subtreeTest = new RewriteRuleSubtreeStream(
				CreateTreeAdaptor(), "RewriteRuleSubtreeStream test");

			subtreeTest.NextTree();
		}

		[Test]
		[ExpectedException(typeof(RewriteEmptyStreamException), "RewriteRuleNodeStream test")]
		public void CheckRRNodeStreamBehaviourWhileEmpty2() {
			RewriteRuleNodeStream nodeTest = new RewriteRuleNodeStream(CreateTreeAdaptor(),
				"RewriteRuleNodeStream test");

			nodeTest.NextTree();
		}

		[Test]
		[ExpectedException(typeof(RewriteEmptyStreamException), "RewriteRuleTokenStream test")]
		public void CheckRRTokenStreamBehaviourWhileEmpty3() {
			RewriteRuleTokenStream tokenTest = new RewriteRuleTokenStream(CreateTreeAdaptor(),
				"RewriteRuleTokenStream test");

			tokenTest.NextToken();
		}

		#endregion

		#region Behaviour with Elements
		[Test]
		[ExpectedException(typeof(RewriteCardinalityException), "RewriteRuleTokenStream test")]
		public void CheckRRTokenStreamBehaviourWithElements() {
			RewriteRuleTokenStream tokenTest = new RewriteRuleTokenStream(CreateTreeAdaptor(),
				"RewriteRuleTokenStream test");

			IToken token1 = CreateToken(1, "test token without any real context");

			// Test Add()
			tokenTest.Add(token1);
			Assert.AreEqual(1, tokenTest.Size(), "tokenTest should have the size 1 (1).");
			Assert.IsTrue(tokenTest.HasNext(), "HasNext() should be true here (1).");

			// Test NextNode()
			CommonTree tree = (CommonTree) tokenTest.NextNode();
			Assert.AreEqual(token1, tree.Token,
				"The returned token should be equal to the given token (1).");
			Assert.AreEqual(1, tokenTest.Size(), "tokenTest should have the size 1 (2).");
			Assert.IsFalse(tokenTest.HasNext(), "HasNext() should be false here (1).");
			tokenTest.Reset();
			Assert.AreEqual(1, tokenTest.Size(), "tokenTest should have the size 1 (3).");
			Assert.IsTrue(tokenTest.HasNext(), "HasNext() should be true here (2).");

			// Test NextToken()
			IToken returnedToken = tokenTest.NextToken();
			Assert.AreEqual(token1, returnedToken,
				"The returned token should be equal to the given token (2).");
			Assert.AreEqual(1, tokenTest.Size(), "tokenTest should have the size 1 (4).");
			Assert.IsFalse(tokenTest.HasNext(), "HasNext() should be false here (2).");
			tokenTest.Reset();
			Assert.AreEqual(1, tokenTest.Size(), "tokenTest should have the size 1 (5).");
			Assert.IsTrue(tokenTest.HasNext(), "HasNext() should be true here (3).");

			// Test NextTree()
			returnedToken = (IToken) tokenTest.NextTree();
			Assert.AreEqual(token1, returnedToken,
				"The returned token should be equal to the given token (3).");
			Assert.AreEqual(1, tokenTest.Size(), "tokenTest should have the size 1 (6).");
			Assert.IsFalse(tokenTest.HasNext(), "HasNext() should be false here (2).");
			tokenTest.Reset();
			Assert.AreEqual(1, tokenTest.Size(), "tokenTest should have the size 1 (7).");
			Assert.IsTrue(tokenTest.HasNext(), "HasNext() should be true here (3).");

			// Test, what happens with two elements
			IToken token2 = CreateToken(2, "test token without any real context");

			tokenTest.Add(token2);
			Assert.AreEqual(2, tokenTest.Size(), "tokenTest should have the size 2 (1).");
			Assert.IsTrue(tokenTest.HasNext(), "HasNext() should be true here (4).");
			returnedToken = tokenTest.NextToken();
			Assert.AreEqual(token1, returnedToken,
				"The returned token should be equal to the given token (4).");
			Assert.AreEqual(2, tokenTest.Size(), "tokenTest should have the size 2 (2).");
			Assert.IsTrue(tokenTest.HasNext(), "HasNext() should be true here (5).");
			returnedToken = tokenTest.NextToken();
			Assert.AreEqual(token2, returnedToken,
				"The returned token should be equal to the given token (5).");
			Assert.IsFalse(tokenTest.HasNext(), "HasNext() should be false here (3).");

			// Test exception
			tokenTest.NextToken();
		}

		[Test]
		[ExpectedException(typeof(RewriteCardinalityException),
			"RewriteRuleSubtreeStream test")]
		public void CheckRRSubtreeStreamBehaviourWithElements() {
			RewriteRuleSubtreeStream subtreeTest =
				new RewriteRuleSubtreeStream(CreateTreeAdaptor(),
				"RewriteRuleSubtreeStream test");

			IToken token1 = CreateToken(1, "test token without any real context");
			ITree tree1 = CreateTree(token1);

			// Test Add()
			subtreeTest.Add(tree1);
			Assert.AreEqual(1, subtreeTest.Size(), "subtreeTest should have the size 1 (1).");
			Assert.IsTrue(subtreeTest.HasNext(), "HasNext() should be true here (1).");

			// Test NextNode()
			Assert.AreEqual(tree1, (ITree) subtreeTest.NextNode(),
				"The returned tree should be equal to the given tree (1).");
			Assert.AreEqual(1, subtreeTest.Size(), "subtreeTest should have the size 1 (2).");
			Assert.IsFalse(subtreeTest.HasNext(), "HasNext() should be false here (1).");
			subtreeTest.Reset();
			Assert.AreEqual(1, subtreeTest.Size(), "subtreeTest should have the size 1 (3).");
			Assert.IsTrue(subtreeTest.HasNext(), "HasNext() should be true here (2).");
			
			// Test NextTree()
			CommonTree returnedTree = (CommonTree) subtreeTest.NextTree();
			Assert.AreEqual(token1, returnedTree.Token,
				"The returned token should be equal to the given token (3).");
			Assert.AreEqual(1, subtreeTest.Size(), "subtreeTest should have the size 1 (4).");
			Assert.IsFalse(subtreeTest.HasNext(), "HasNext() should be false here (2).");
			subtreeTest.Reset();
			Assert.AreEqual(1, subtreeTest.Size(), "subtreeTest should have the size 1 (5).");
			Assert.IsTrue(subtreeTest.HasNext(), "HasNext() should be true here (3).");
			
			// Test, what happens with two elements
			IToken token2 = CreateToken(2, "test token without any real context");
			ITree tree2 = CreateTree(token2);

			subtreeTest.Add(tree2);
			Assert.AreEqual(2, subtreeTest.Size(), "subtreeTest should have the size 2 (1).");
			Assert.IsTrue(subtreeTest.HasNext(), "HasNext() should be true here (4).");
			returnedTree = (CommonTree) subtreeTest.NextTree();
			Assert.AreEqual(token1, returnedTree.Token,
				"The returned token should be equal to the given token (4).");
			Assert.AreEqual(2, subtreeTest.Size(), "subtreeTest should have the size 2 (2).");
			Assert.IsTrue(subtreeTest.HasNext(), "HasNext() should be true here (5).");
			returnedTree = (CommonTree) subtreeTest.NextTree();
			Assert.AreEqual(token2, returnedTree.Token,
				"The returned token should be equal to the given token (5).");
			Assert.IsFalse(subtreeTest.HasNext(), "HasNext() should be false here (3).");

			// Test exception
			subtreeTest.NextTree();
		}

		[Test]
		[ExpectedException(typeof(RewriteCardinalityException), "RewriteRuleNodeStream test")]
		public void CheckRRNodeStreamBehaviourWithElements() {
			RewriteRuleNodeStream nodeTest = new RewriteRuleNodeStream(CreateTreeAdaptor(),
				"RewriteRuleNodeStream test");

			IToken token1 = CreateToken(1, "test token without any real context");
			ITree tree1 = CreateTree(token1);

			// Test Add()
			nodeTest.Add(tree1);
			Assert.AreEqual(1, nodeTest.Size(), "nodeTest should have the size 1 (1).");
			Assert.IsTrue(nodeTest.HasNext(), "HasNext() should be true here (1).");

			// Test NextNode()
			CommonTree returnedTree = (CommonTree) nodeTest.NextNode();
			Assert.AreEqual(tree1.Type, returnedTree.Type,
				"The returned tree should be equal to the given tree (1).");
			Assert.AreEqual(1, nodeTest.Size(), "nodeTest should have the size 1 (2).");
			Assert.IsFalse(nodeTest.HasNext(), "HasNext() should be false here (1).");
			nodeTest.Reset();
			Assert.AreEqual(1, nodeTest.Size(), "nodeTest should have the size 1 (3).");
			Assert.IsTrue(nodeTest.HasNext(), "HasNext() should be true here (2).");

			// Test NextTree()
			returnedTree = (CommonTree) nodeTest.NextTree();
			Assert.AreEqual(token1, returnedTree.Token,
				"The returned token should be equal to the given token (3).");
			Assert.AreEqual(1, nodeTest.Size(), "nodeTest should have the size 1 (4).");
			Assert.IsFalse(nodeTest.HasNext(), "HasNext() should be false here (2).");
			nodeTest.Reset();
			Assert.AreEqual(1, nodeTest.Size(), "nodeTest should have the size 1 (5).");
			Assert.IsTrue(nodeTest.HasNext(), "HasNext() should be true here (3).");

			// Test, what happens with two elements
			IToken token2 = CreateToken(2, "test token without any real context");
			ITree tree2 = CreateTree(token2);

			nodeTest.Add(tree2);
			Assert.AreEqual(2, nodeTest.Size(), "nodeTest should have the size 2 (1).");
			Assert.IsTrue(nodeTest.HasNext(), "HasNext() should be true here (4).");
			returnedTree = (CommonTree) nodeTest.NextTree();
			Assert.AreEqual(token1, returnedTree.Token,
				"The returned token should be equal to the given token (4).");
			Assert.AreEqual(2, nodeTest.Size(), "nodeTest should have the size 2 (2).");
			Assert.IsTrue(nodeTest.HasNext(), "HasNext() should be true here (5).");
			returnedTree = (CommonTree) nodeTest.NextTree();
			Assert.AreEqual(token2, returnedTree.Token,
				"The returned token should be equal to the given token (5).");
			Assert.IsFalse(nodeTest.HasNext(), "HasNext() should be false here (3).");

			// Test exception
			nodeTest.NextTree();
		}

		#endregion

		#endregion


		#region Helper Methods

		private ITreeAdaptor CreateTreeAdaptor() {
			return new CommonTreeAdaptor();
		}

		private ITree CreateTree(IToken token) {
			return new CommonTree(token);
		}

		private IToken CreateToken(int type, string text) {
			return new CommonToken(type, text);
		}

		private IList<IToken> CreateTokenList(int count) {
			IList<IToken> list = new List<IToken>();
			for (int i = 0; i < count; i++) {
				list.Add(new CommonToken((i+1), "test token " + (i+1).ToString() +
					" without any real context"));
			}
			return list;
		}

		#endregion
	}
}