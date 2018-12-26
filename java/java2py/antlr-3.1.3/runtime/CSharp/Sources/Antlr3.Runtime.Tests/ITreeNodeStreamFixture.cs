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
	using ITreeNodeStream = Antlr.Runtime.Tree.ITreeNodeStream;
	using CommonTree = Antlr.Runtime.Tree.CommonTree;
	using CommonTreeNodeStream = Antlr.Runtime.Tree.CommonTreeNodeStream;
	using UnBufferedTreeNodeStream = Antlr.Runtime.Tree.UnBufferedTreeNodeStream;

	using MbUnit.Framework;

	[TestFixture]
	public class ITreeNodeStreamFixture : TestFixtureBase
	{
		#region CommonTreeNodeStream Tests

		[Test]
		public void testSingleNode()
		{
			ITree t = new CommonTree(new CommonToken(101));

			ITreeNodeStream stream = CreateCommonTreeNodeStream(t);
			string expected = " 101";
			string actual = GetStringOfEntireStreamContentsWithNodeTypesOnly(stream);
			Assert.AreEqual(expected, actual);

			expected = " 101";
			actual = stream.ToString();
			Assert.AreEqual(expected, actual);
		}

		[Test]
		/// <summary>
		/// Test a tree with four nodes - ^(101 ^(102 103) 104)
		/// </summary>
		public void test4Nodes()
		{
			ITree t = new CommonTree(new CommonToken(101));
			t.AddChild(new CommonTree(new CommonToken(102)));
			t.GetChild(0).AddChild(new CommonTree(new CommonToken(103)));
			t.AddChild(new CommonTree(new CommonToken(104)));

			ITreeNodeStream stream = CreateCommonTreeNodeStream(t);
			string expected = " 101 102 103 104";
			string actual = GetStringOfEntireStreamContentsWithNodeTypesOnly(stream);
			Assert.AreEqual(expected, actual);

			expected = " 101 2 102 2 103 3 104 3";
			actual = stream.ToString();
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testList()
		{
			ITree root = new CommonTree((IToken)null);

			ITree t = new CommonTree(new CommonToken(101));
			t.AddChild(new CommonTree(new CommonToken(102)));
			t.GetChild(0).AddChild(new CommonTree(new CommonToken(103)));
			t.AddChild(new CommonTree(new CommonToken(104)));

			ITree u = new CommonTree(new CommonToken(105));

			root.AddChild(t);
			root.AddChild(u);

			CommonTreeNodeStream stream = new CommonTreeNodeStream(root);
			string expected = " 101 102 103 104 105";
			string actual = GetStringOfEntireStreamContentsWithNodeTypesOnly(stream);
			Assert.AreEqual(expected, actual);

			expected = " 101 2 102 2 103 3 104 3 105";
			actual = stream.ToString();
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testFlatList()
		{
			ITree root = new CommonTree((IToken)null);

			root.AddChild(new CommonTree(new CommonToken(101)));
			root.AddChild(new CommonTree(new CommonToken(102)));
			root.AddChild(new CommonTree(new CommonToken(103)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(root);
			string expected = " 101 102 103";
			string actual = GetStringOfEntireStreamContentsWithNodeTypesOnly(stream);
			Assert.AreEqual(expected, actual);

			expected = " 101 102 103";
			actual = stream.ToString();
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testListWithOneNode()
		{
			ITree root = new CommonTree((IToken)null);

			root.AddChild(new CommonTree(new CommonToken(101)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(root);
			string expected = " 101";
			string actual = GetStringOfEntireStreamContentsWithNodeTypesOnly(stream);
			Assert.AreEqual(expected, actual);

			expected = " 101";
			actual = stream.ToString();
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testAoverB()
		{
			ITree t = new CommonTree(new CommonToken(101));
			t.AddChild(new CommonTree(new CommonToken(102)));

			ITreeNodeStream stream = CreateCommonTreeNodeStream(t);
			string expected = " 101 102";
			string actual = GetStringOfEntireStreamContentsWithNodeTypesOnly(stream);
			Assert.AreEqual(expected, actual);

			expected = " 101 2 102 3";
			actual = stream.ToString();
			Assert.AreEqual(expected, actual);
		}

		[Test]
		public void testLT()
		{
			// ^(101 ^(102 103) 104)
			ITree t = new CommonTree(new CommonToken(101));
			t.AddChild(new CommonTree(new CommonToken(102)));
			t.GetChild(0).AddChild(new CommonTree(new CommonToken(103)));
			t.AddChild(new CommonTree(new CommonToken(104)));

			ITreeNodeStream stream = CreateCommonTreeNodeStream(t);
			Assert.AreEqual(101, ((ITree)stream.LT(1)).Type);
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(2)).Type);
			Assert.AreEqual(102, ((ITree)stream.LT(3)).Type);
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(4)).Type);
			Assert.AreEqual(103, ((ITree)stream.LT(5)).Type);
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(6)).Type);
			Assert.AreEqual(104, ((ITree)stream.LT(7)).Type);
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(8)).Type);
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(9)).Type);
			// check way ahead
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(100)).Type);
		}

		[Test]
		public void testMarkRewindEntire()
		{
			// ^(101 ^(102 103 ^(106 107) ) 104 105)
			// stream has 7 real + 6 nav nodes
			// Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r0.AddChild(r1);
			r1.AddChild(new CommonTree(new CommonToken(103)));
			ITree r2 = new CommonTree(new CommonToken(106));
			r2.AddChild(new CommonTree(new CommonToken(107)));
			r1.AddChild(r2);
			r0.AddChild(new CommonTree(new CommonToken(104)));
			r0.AddChild(new CommonTree(new CommonToken(105)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);
			int m = stream.Mark(); // MARK
			for (int k = 1; k <= 13; k++)
			{ // consume til end
				stream.LT(1);
				stream.Consume();
			}
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(1)).Type);
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(-1)).Type);
			stream.Rewind(m);      // REWIND

			// consume til end again :)
			for (int k = 1; k <= 13; k++)
			{ // consume til end
				stream.LT(1);
				stream.Consume();
			}
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(1)).Type);
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(-1)).Type);
		}

		[Test]
		public void testMarkRewindInMiddle()
		{
			// ^(101 ^(102 103 ^(106 107) ) 104 105)
			// stream has 7 real + 6 nav nodes
			// Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r0.AddChild(r1);
			r1.AddChild(new CommonTree(new CommonToken(103)));
			ITree r2 = new CommonTree(new CommonToken(106));
			r2.AddChild(new CommonTree(new CommonToken(107)));
			r1.AddChild(r2);
			r0.AddChild(new CommonTree(new CommonToken(104)));
			r0.AddChild(new CommonTree(new CommonToken(105)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);
			for (int k = 1; k <= 7; k++)
			{ // consume til middle
				//System.out.println(((ITree)stream.LT(1)).Type);
				stream.Consume();
			}
			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
			int m = stream.Mark(); // MARK
			stream.Consume(); // consume 107
			stream.Consume(); // consume UP
			stream.Consume(); // consume UP
			stream.Consume(); // consume 104
			stream.Rewind(m);      // REWIND

			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(104, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			// now we're past rewind position
			Assert.AreEqual(105, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(1)).Type);
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(-1)).Type);
		}

		[Test]
		public void testMarkRewindNested()
		{
			// ^(101 ^(102 103 ^(106 107) ) 104 105)
			// stream has 7 real + 6 nav nodes
			// Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r0.AddChild(r1);
			r1.AddChild(new CommonTree(new CommonToken(103)));
			ITree r2 = new CommonTree(new CommonToken(106));
			r2.AddChild(new CommonTree(new CommonToken(107)));
			r1.AddChild(r2);
			r0.AddChild(new CommonTree(new CommonToken(104)));
			r0.AddChild(new CommonTree(new CommonToken(105)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);
			int m = stream.Mark(); // MARK at start
			stream.Consume(); // consume 101
			stream.Consume(); // consume DN
			int m2 = stream.Mark(); // MARK on 102
			stream.Consume(); // consume 102
			stream.Consume(); // consume DN
			stream.Consume(); // consume 103
			stream.Consume(); // consume 106
			stream.Rewind(m2);      // REWIND to 102
			Assert.AreEqual(102, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			// stop at 103 and rewind to start
			stream.Rewind(m); // REWIND to 101
			Assert.AreEqual(101, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(102, ((ITree)stream.LT(1)).Type);
			stream.Consume();
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
		}

		[Test]
		public void testSeek()
		{
			// ^(101 ^(102 103 ^(106 107) ) 104 105)
			// stream has 7 real + 6 nav nodes
			// Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r0.AddChild(r1);
			r1.AddChild(new CommonTree(new CommonToken(103)));
			ITree r2 = new CommonTree(new CommonToken(106));
			r2.AddChild(new CommonTree(new CommonToken(107)));
			r1.AddChild(r2);
			r0.AddChild(new CommonTree(new CommonToken(104)));
			r0.AddChild(new CommonTree(new CommonToken(105)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);
			stream.Consume(); // consume 101
			stream.Consume(); // consume DN
			stream.Consume(); // consume 102
			stream.Seek(7);   // seek to 107
			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 107
			stream.Consume(); // consume UP
			stream.Consume(); // consume UP
			Assert.AreEqual(104, ((ITree)stream.LT(1)).Type);
		}

		[Test]
		public void testSeekFromStart()
		{
			// ^(101 ^(102 103 ^(106 107) ) 104 105)
			// stream has 7 real + 6 nav nodes
			// Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r0.AddChild(r1);
			r1.AddChild(new CommonTree(new CommonToken(103)));
			ITree r2 = new CommonTree(new CommonToken(106));
			r2.AddChild(new CommonTree(new CommonToken(107)));
			r1.AddChild(r2);
			r0.AddChild(new CommonTree(new CommonToken(104)));
			r0.AddChild(new CommonTree(new CommonToken(105)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);
			stream.Seek(7);   // seek to 107
			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 107
			stream.Consume(); // consume UP
			stream.Consume(); // consume UP
			Assert.AreEqual(104, ((ITree)stream.LT(1)).Type);
		}

		[Test]
		public void testPushPop()
		{
			// ^(101 ^(102 103) ^(104 105) ^(106 107) 108 109)
			// stream has 9 real + 8 nav nodes
			// Sequence of types: 101 DN 102 DN 103 UP 104 DN 105 UP 106 DN 107 UP 108 109 UP
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r1.AddChild(new CommonTree(new CommonToken(103)));
			r0.AddChild(r1);
			ITree r2 = new CommonTree(new CommonToken(104));
			r2.AddChild(new CommonTree(new CommonToken(105)));
			r0.AddChild(r2);
			ITree r3 = new CommonTree(new CommonToken(106));
			r3.AddChild(new CommonTree(new CommonToken(107)));
			r0.AddChild(r3);
			r0.AddChild(new CommonTree(new CommonToken(108)));
			r0.AddChild(new CommonTree(new CommonToken(109)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);
			String expecting = " 101 2 102 2 103 3 104 2 105 3 106 2 107 3 108 109 3";
			String found = stream.ToString();
			Assert.AreEqual(expecting, found);

			// Assume we want to hit node 107 and then "call 102" then return

			int indexOf102 = 2;
			int indexOf107 = 12;
			for (int k = 1; k <= indexOf107; k++)
			{ // consume til 107 node
				stream.Consume();
			}
			// CALL 102
			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
			stream.Push(indexOf102);
			Assert.AreEqual(102, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 102
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume DN
			Assert.AreEqual(103, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 103
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			// RETURN
			stream.Pop();
			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
		}

		[Test]
		public void testNestedPushPop()
		{
			// ^(101 ^(102 103) ^(104 105) ^(106 107) 108 109)
			// stream has 9 real + 8 nav nodes
			// Sequence of types: 101 DN 102 DN 103 UP 104 DN 105 UP 106 DN 107 UP 108 109 UP
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r1.AddChild(new CommonTree(new CommonToken(103)));
			r0.AddChild(r1);
			ITree r2 = new CommonTree(new CommonToken(104));
			r2.AddChild(new CommonTree(new CommonToken(105)));
			r0.AddChild(r2);
			ITree r3 = new CommonTree(new CommonToken(106));
			r3.AddChild(new CommonTree(new CommonToken(107)));
			r0.AddChild(r3);
			r0.AddChild(new CommonTree(new CommonToken(108)));
			r0.AddChild(new CommonTree(new CommonToken(109)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);

			// Assume we want to hit node 107 and then "call 102", which
			// calls 104, then return

			int indexOf102 = 2;
			int indexOf107 = 12;
			for (int k = 1; k <= indexOf107; k++)
			{ // consume til 107 node
				stream.Consume();
			}
			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
			// CALL 102
			stream.Push(indexOf102);
			Assert.AreEqual(102, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 102
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume DN
			Assert.AreEqual(103, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 103

			// CALL 104
			int indexOf104 = 6;
			stream.Push(indexOf104);
			Assert.AreEqual(104, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 102
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume DN
			Assert.AreEqual(105, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 103
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			// RETURN (to UP node in 102 subtree)
			stream.Pop();

			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			// RETURN (to empty stack)
			stream.Pop();
			Assert.AreEqual(107, ((ITree)stream.LT(1)).Type);
		}

		[Test]
		public void testPushPopFromEOF()
		{
			// ^(101 ^(102 103) ^(104 105) ^(106 107) 108 109)
			// stream has 9 real + 8 nav nodes
			// Sequence of types: 101 DN 102 DN 103 UP 104 DN 105 UP 106 DN 107 UP 108 109 UP
			ITree r0 = new CommonTree(new CommonToken(101));
			ITree r1 = new CommonTree(new CommonToken(102));
			r1.AddChild(new CommonTree(new CommonToken(103)));
			r0.AddChild(r1);
			ITree r2 = new CommonTree(new CommonToken(104));
			r2.AddChild(new CommonTree(new CommonToken(105)));
			r0.AddChild(r2);
			ITree r3 = new CommonTree(new CommonToken(106));
			r3.AddChild(new CommonTree(new CommonToken(107)));
			r0.AddChild(r3);
			r0.AddChild(new CommonTree(new CommonToken(108)));
			r0.AddChild(new CommonTree(new CommonToken(109)));

			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);

			while (stream.LA(1) != Token.EOF)
			{
				stream.Consume();
			}
			int indexOf102 = 2;
			int indexOf104 = 6;
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(1)).Type);

			// CALL 102
			stream.Push(indexOf102);
			Assert.AreEqual(102, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 102
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume DN
			Assert.AreEqual(103, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 103
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			// RETURN (to empty stack)
			stream.Pop();
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(1)).Type);

			// CALL 104
			stream.Push(indexOf104);
			Assert.AreEqual(104, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 102
			Assert.AreEqual(Token.DOWN, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume DN
			Assert.AreEqual(105, ((ITree)stream.LT(1)).Type);
			stream.Consume(); // consume 103
			Assert.AreEqual(Token.UP, ((ITree)stream.LT(1)).Type);
			// RETURN (to empty stack)
			stream.Pop();
			Assert.AreEqual(Token.EOF, ((ITree)stream.LT(1)).Type);
		}

		[Test]
		public void testStackStretch()
		{
			// make more than INITIAL_CALL_STACK_SIZE pushes
			ITree r0 = new CommonTree(new CommonToken(101));
			CommonTreeNodeStream stream = new CommonTreeNodeStream(r0);
			// go 1 over initial size
			for (int i = 1; i <= CommonTreeNodeStream.INITIAL_CALL_STACK_SIZE + 1; i++)
			{
				stream.Push(i);
			}
			Assert.AreEqual(10, stream.Pop());
			Assert.AreEqual(9, stream.Pop());
		}

		#endregion


		#region UnBufferedTreeNodeStream Tests

		[Test]
		public void testBufferOverflow()
		{
			StringBuilder buf = new StringBuilder();
			StringBuilder buf2 = new StringBuilder();
			// make ^(101 102 ... n)
			ITree t = new CommonTree(new CommonToken(101));
			buf.Append(" 101");
			buf2.Append(" 101");
			buf2.Append(" ");
			buf2.Append(Token.DOWN);
			for (int i = 0; i <= UnBufferedTreeNodeStream.INITIAL_LOOKAHEAD_BUFFER_SIZE + 10; i++)
			{
				t.AddChild(new CommonTree(new CommonToken(102 + i)));
				buf.Append(" ");
				buf.Append(102 + i);
				buf2.Append(" ");
				buf2.Append(102 + i);
			}
			buf2.Append(" ");
			buf2.Append(Token.UP);

			ITreeNodeStream stream = CreateUnBufferedTreeNodeStream(t);
			String expecting = buf.ToString();
			String found = GetStringOfEntireStreamContentsWithNodeTypesOnly(stream);
			Assert.AreEqual(expecting, found);

			expecting = buf2.ToString();
			found = stream.ToString();
			Assert.AreEqual(expecting, found);
		}

		/// <summary>
		/// Test what happens when tail hits the end of the buffer, but there
		/// is more room left.
		/// </summary>
		/// <remarks>
		/// Specifically that would mean that head is not at 0 but has 
		/// advanced somewhere to the middle of the lookahead buffer.
		/// 
		/// Use Consume() to advance N nodes into lookahead.  Then use LT()
		/// to load at least INITIAL_LOOKAHEAD_BUFFER_SIZE-N nodes so the
		/// buffer has to wrap.
		/// </remarks>
		[Test]
		public void testBufferWrap()
		{
			int N = 10;
			// make tree with types: 1 2 ... INITIAL_LOOKAHEAD_BUFFER_SIZE+N
			ITree t = new CommonTree((IToken)null);
			for (int i = 0; i < UnBufferedTreeNodeStream.INITIAL_LOOKAHEAD_BUFFER_SIZE + N; i++)
			{
				t.AddChild(new CommonTree(new CommonToken(i + 1)));
			}

			// move head to index N
			ITreeNodeStream stream = CreateUnBufferedTreeNodeStream(t);
			for (int i = 1; i <= N; i++)
			{ // consume N
				ITree node = (ITree)stream.LT(1);
				Assert.AreEqual(i, node.Type);
				stream.Consume();
			}

			// now use LT to lookahead past end of buffer
			int remaining = UnBufferedTreeNodeStream.INITIAL_LOOKAHEAD_BUFFER_SIZE - N;
			int wrapBy = 4; // wrap around by 4 nodes
			Assert.IsTrue(wrapBy < N, "bad test code; wrapBy must be less than N");
			for (int i = 1; i <= remaining + wrapBy; i++)
			{ // wrap past end of buffer
				ITree node = (ITree)stream.LT(i); // look ahead to ith token
				Assert.AreEqual(N + i, node.Type);
			}
		}

		#endregion


		#region Helper Methods

		protected ITreeNodeStream CreateCommonTreeNodeStream(object t)
		{
			return new CommonTreeNodeStream(t);
		}

		protected ITreeNodeStream CreateUnBufferedTreeNodeStream(object t)
		{
			return new UnBufferedTreeNodeStream(t);
		}

		public string GetStringOfEntireStreamContentsWithNodeTypesOnly(ITreeNodeStream nodes)
		{
			StringBuilder buf = new StringBuilder();
			for (int i = 0; i < nodes.Count; i++)
			{
				object t = nodes.LT(i + 1);
				int type = nodes.TreeAdaptor.GetNodeType(t);
				if (!((type == Token.DOWN) || (type == Token.UP)))
				{
					buf.Append(" ");
					buf.Append(type);
				}
			}
			return buf.ToString();
		}

		#endregion
	}
}