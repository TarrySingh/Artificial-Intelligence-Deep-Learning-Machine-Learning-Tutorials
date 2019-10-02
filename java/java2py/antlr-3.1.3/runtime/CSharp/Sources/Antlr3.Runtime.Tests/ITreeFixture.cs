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
	using StringBuilder = System.Text.StringBuilder;

	using IToken = Antlr.Runtime.IToken;
	using Token = Antlr.Runtime.Token;
	using CommonToken = Antlr.Runtime.CommonToken;
	using ITree = Antlr.Runtime.Tree.ITree;
	using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;
	using CommonTree = Antlr.Runtime.Tree.CommonTree;
	using CommonTreeAdaptor = Antlr.Runtime.Tree.CommonTreeAdaptor;

	using MbUnit.Framework;

	[TestFixture]
	public class ITreeFixture : TestFixtureBase
	{
		#region CommonTree Tests

		[Test]
		public void testSingleNode()
		{
			CommonTree t = new CommonTree(new CommonToken(101));
			Assert.IsNull(t.parent);
			Assert.AreEqual(-1, t.childIndex);
		}

		[Test]
		public void test4Nodes()
		{
			// ^(101 ^(102 103) 104)
			CommonTree r0 = new CommonTree(new CommonToken(101));
			r0.AddChild(new CommonTree(new CommonToken(102)));
			r0.GetChild(0).AddChild(new CommonTree(new CommonToken(103)));
			r0.AddChild(new CommonTree(new CommonToken(104)));

			Assert.IsNull(r0.parent);
			Assert.AreEqual(-1, r0.childIndex);
		}

		[Test]
		public void testList()
		{
			// ^(nil 101 102 103)
			CommonTree r0 = new CommonTree((IToken)null);
			CommonTree c0, c1, c2;
			r0.AddChild(c0 = new CommonTree(new CommonToken(101)));
			r0.AddChild(c1 = new CommonTree(new CommonToken(102)));
			r0.AddChild(c2 = new CommonTree(new CommonToken(103)));

			Assert.IsNull(r0.parent);
			Assert.AreEqual(-1, r0.childIndex);
			Assert.AreEqual(r0, c0.parent);
			Assert.AreEqual(0, c0.childIndex);
			Assert.AreEqual(r0, c1.parent);
			Assert.AreEqual(1, c1.childIndex);
			Assert.AreEqual(r0, c2.parent);
			Assert.AreEqual(2, c2.childIndex);
		}

		[Test]
		public void testList2()
		{
			// Add child ^(nil 101 102 103) to root 5
			// should pull 101 102 103 directly to become 5's child list
			CommonTree root = new CommonTree(new CommonToken(5));

			// child tree
			CommonTree r0 = new CommonTree((IToken)null);
			CommonTree c0, c1, c2;
			r0.AddChild(c0 = new CommonTree(new CommonToken(101)));
			r0.AddChild(c1 = new CommonTree(new CommonToken(102)));
			r0.AddChild(c2 = new CommonTree(new CommonToken(103)));

			root.AddChild(r0);

			Assert.IsNull(root.parent);
			Assert.AreEqual(-1, root.childIndex);
			// check children of root all point at root
			Assert.AreEqual(root, c0.parent);
			Assert.AreEqual(0, c0.childIndex);
			Assert.AreEqual(root, c0.parent);
			Assert.AreEqual(1, c1.childIndex);
			Assert.AreEqual(root, c0.parent);
			Assert.AreEqual(2, c2.childIndex);
		}

		[Test]
		public void testAddListToExistChildren()
		{
			// Add child ^(nil 101 102 103) to root ^(5 6)
			// should add 101 102 103 to end of 5's child list
			CommonTree root = new CommonTree(new CommonToken(5));
			root.AddChild(new CommonTree(new CommonToken(6)));

			// child tree
			CommonTree r0 = new CommonTree((IToken)null);
			CommonTree c0, c1, c2;
			r0.AddChild(c0 = new CommonTree(new CommonToken(101)));
			r0.AddChild(c1 = new CommonTree(new CommonToken(102)));
			r0.AddChild(c2 = new CommonTree(new CommonToken(103)));

			root.AddChild(r0);

			Assert.IsNull(root.parent);
			Assert.AreEqual(-1, root.childIndex);
			// check children of root all point at root
			Assert.AreEqual(root, c0.parent);
			Assert.AreEqual(1, c0.childIndex);
			Assert.AreEqual(root, c0.parent);
			Assert.AreEqual(2, c1.childIndex);
			Assert.AreEqual(root, c0.parent);
			Assert.AreEqual(3, c2.childIndex);
		}

		[Test]
		public void testDupTree()
		{
			// ^(101 ^(102 103 ^(106 107) ) 104 105)
			CommonTree r0 = new CommonTree(new CommonToken(101));
			CommonTree r1 = new CommonTree(new CommonToken(102));
			r0.AddChild(r1);
			r1.AddChild(new CommonTree(new CommonToken(103)));
			ITree r2 = new CommonTree(new CommonToken(106));
			r2.AddChild(new CommonTree(new CommonToken(107)));
			r1.AddChild(r2);
			r0.AddChild(new CommonTree(new CommonToken(104)));
			r0.AddChild(new CommonTree(new CommonToken(105)));

			CommonTree dup = (CommonTree)(new CommonTreeAdaptor()).DupTree(r0);

			Assert.IsNull(dup.parent);
			Assert.AreEqual(-1, dup.childIndex);
			dup.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testBecomeRoot()
		{
			// 5 becomes new root of ^(nil 101 102 103)
			CommonTree newRoot = new CommonTree(new CommonToken(5));

			CommonTree oldRoot = new CommonTree((IToken)null);
			oldRoot.AddChild(new CommonTree(new CommonToken(101)));
			oldRoot.AddChild(new CommonTree(new CommonToken(102)));
			oldRoot.AddChild(new CommonTree(new CommonToken(103)));

			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			adaptor.BecomeRoot(newRoot, oldRoot);
			newRoot.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testBecomeRoot2()
		{
			// 5 becomes new root of ^(101 102 103)
			CommonTree newRoot = new CommonTree(new CommonToken(5));

			CommonTree oldRoot = new CommonTree(new CommonToken(101));
			oldRoot.AddChild(new CommonTree(new CommonToken(102)));
			oldRoot.AddChild(new CommonTree(new CommonToken(103)));

			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			adaptor.BecomeRoot(newRoot, oldRoot);
			newRoot.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testBecomeRoot3()
		{
			// ^(nil 5) becomes new root of ^(nil 101 102 103)
			CommonTree newRoot = new CommonTree((IToken)null);
			newRoot.AddChild(new CommonTree(new CommonToken(5)));

			CommonTree oldRoot = new CommonTree((IToken)null);
			oldRoot.AddChild(new CommonTree(new CommonToken(101)));
			oldRoot.AddChild(new CommonTree(new CommonToken(102)));
			oldRoot.AddChild(new CommonTree(new CommonToken(103)));

			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			adaptor.BecomeRoot(newRoot, oldRoot);
			newRoot.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testBecomeRoot5()
		{
			// ^(nil 5) becomes new root of ^(101 102 103)
			CommonTree newRoot = new CommonTree((IToken)null);
			newRoot.AddChild(new CommonTree(new CommonToken(5)));

			CommonTree oldRoot = new CommonTree(new CommonToken(101));
			oldRoot.AddChild(new CommonTree(new CommonToken(102)));
			oldRoot.AddChild(new CommonTree(new CommonToken(103)));

			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			adaptor.BecomeRoot(newRoot, oldRoot);
			newRoot.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testBecomeRoot6()
		{
			// emulates construction of ^(5 6)
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			CommonTree root_0 = (CommonTree)adaptor.GetNilNode();
			CommonTree root_1 = (CommonTree)adaptor.GetNilNode();
			root_1 = (CommonTree)adaptor.BecomeRoot(new CommonTree(new CommonToken(5)), root_1);

			adaptor.AddChild(root_1, new CommonTree(new CommonToken(6)));

			adaptor.AddChild(root_0, root_1);

			root_0.SanityCheckParentAndChildIndexes();
		}

		// Test replaceChildren

		[Test]
		public void testReplaceWithNoChildren()
		{
			CommonTree t = new CommonTree(new CommonToken(101));
			CommonTree newChild = new CommonTree(new CommonToken(5));
			bool error = false;
			try
			{
				t.ReplaceChildren(0, 0, newChild);
			}
			catch (Exception)
			{
				error = true;
			}
			Assert.IsTrue(error);
		}

		[Test]
		public void testReplaceWithOneChildren()
		{
			// assume token type 99 and use text
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			CommonTree c0 = new CommonTree(new CommonToken(99, "b"));
			t.AddChild(c0);

			CommonTree newChild = new CommonTree(new CommonToken(99, "c"));
			t.ReplaceChildren(0, 0, newChild);
			String expected = "(a c)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceInMiddle()
		{
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c"))); // index 1
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChild = new CommonTree(new CommonToken(99, "x"));
			t.ReplaceChildren(1, 1, newChild);
			String expected = "(a b x d)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceAtLeft()
		{
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b"))); // index 0
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChild = new CommonTree(new CommonToken(99, "x"));
			t.ReplaceChildren(0, 0, newChild);
			String expected = "(a x c d)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceAtRight()
		{
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d"))); // index 2

			CommonTree newChild = new CommonTree(new CommonToken(99, "x"));
			t.ReplaceChildren(2, 2, newChild);
			String expected = "(a b c x)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceOneWithTwoAtLeft()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChildren = (CommonTree)adaptor.GetNilNode();
			newChildren.AddChild(new CommonTree(new CommonToken(99, "x")));
			newChildren.AddChild(new CommonTree(new CommonToken(99, "y")));

			t.ReplaceChildren(0, 0, newChildren);
			String expected = "(a x y c d)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceOneWithTwoAtRight()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChildren = (CommonTree)adaptor.GetNilNode();
			newChildren.AddChild(new CommonTree(new CommonToken(99, "x")));
			newChildren.AddChild(new CommonTree(new CommonToken(99, "y")));

			t.ReplaceChildren(2, 2, newChildren);
			String expected = "(a b c x y)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceOneWithTwoInMiddle()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChildren = (CommonTree)adaptor.GetNilNode();
			newChildren.AddChild(new CommonTree(new CommonToken(99, "x")));
			newChildren.AddChild(new CommonTree(new CommonToken(99, "y")));

			t.ReplaceChildren(1, 1, newChildren);
			String expected = "(a b x y d)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceTwoWithOneAtLeft()
		{
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChild = new CommonTree(new CommonToken(99, "x"));

			t.ReplaceChildren(0, 1, newChild);
			String expected = "(a x d)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceTwoWithOneAtRight()
		{
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChild = new CommonTree(new CommonToken(99, "x"));

			t.ReplaceChildren(1, 2, newChild);
			String expected = "(a b x)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceAllWithOne()
		{
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChild = new CommonTree(new CommonToken(99, "x"));

			t.ReplaceChildren(0, 2, newChild);
			String expected = "(a x)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		[Test]
		public void testReplaceAllWithTwo()
		{
			ITreeAdaptor adaptor = new CommonTreeAdaptor();
			CommonTree t = new CommonTree(new CommonToken(99, "a"));
			t.AddChild(new CommonTree(new CommonToken(99, "b")));
			t.AddChild(new CommonTree(new CommonToken(99, "c")));
			t.AddChild(new CommonTree(new CommonToken(99, "d")));

			CommonTree newChildren = (CommonTree)adaptor.GetNilNode();
			newChildren.AddChild(new CommonTree(new CommonToken(99, "x")));
			newChildren.AddChild(new CommonTree(new CommonToken(99, "y")));

			t.ReplaceChildren(0, 2, newChildren);
			String expected = "(a x y)";
			Assert.AreEqual(expected, t.ToStringTree());
			t.SanityCheckParentAndChildIndexes();
		}

		#endregion
	}
}