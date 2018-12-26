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


#pragma warning disable 219 // No unused variable warnings

namespace Antlr.Runtime.Tests
{
	using System;
	using Stream = System.IO.Stream;
	using FileStream = System.IO.FileStream;
	using MemoryStream = System.IO.MemoryStream;
	using FileMode = System.IO.FileMode;
	using Encoding = System.Text.Encoding;
	using Encoder = System.Text.Encoder;

	using ANTLRInputStream = Antlr.Runtime.ANTLRInputStream;

	using MbUnit.Framework;

	[TestFixture]
	public class ANTLRxxxxStreamFixture : TestFixtureBase
	{
		private static readonly string grammarStr = ""
				+ "parser grammar p;" + NL
				+ "prog : WHILE ID LCURLY (assign)* RCURLY EOF;" + NL
				+ "assign : ID ASSIGN expr SEMI ;" + NL
				+ "expr : INT | FLOAT | ID ;" + NL;


		#region ANTLRInputStream Tests

		[Test]
		public void TestANTLRInputStreamConstructorDoesNotHang()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] grammarStrBuffer = encoding.GetBytes(grammarStr);
			MemoryStream grammarStream = new MemoryStream(grammarStrBuffer);

			ANTLRInputStream input = new ANTLRInputStream(grammarStream, encoding);
		}

		[Test]
		public void TestSizeOnEmptyANTLRInputStream()
		{
			MemoryStream grammarStream = new MemoryStream(new byte[] { });

			ANTLRInputStream inputStream = new ANTLRInputStream(grammarStream, Encoding.Unicode);
			Assert.AreEqual(0, inputStream.Count);
		}

		[Test]
		public void TestSizeOnANTLRInputStream()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] grammarStrBuffer = encoding.GetBytes(grammarStr);
			MemoryStream grammarStream = new MemoryStream(grammarStrBuffer);

			ANTLRInputStream inputStream = new ANTLRInputStream(grammarStream, Encoding.Unicode);
			Assert.AreEqual(grammarStr.Length, inputStream.Count);
		}

		[Test]
		public void TestConsumeAndIndexOnANTLRInputStream()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] grammarStrBuffer = encoding.GetBytes(grammarStr);
			MemoryStream grammarStream = new MemoryStream(grammarStrBuffer);

			ANTLRInputStream inputStream = new ANTLRInputStream(grammarStream, Encoding.Unicode);
			Assert.AreEqual(0, inputStream.Index());

			inputStream.Consume();
			Assert.AreEqual(1, inputStream.Index());

			inputStream.Consume();
			Assert.AreEqual(2, inputStream.Index());

			while (inputStream.Index() < inputStream.Count)
			{
				inputStream.Consume();
			}
			Assert.AreEqual(inputStream.Index(), inputStream.Count);
		}

		[Test]
		public void TestConsumeAllCharactersInAnANTLRInputStream()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] grammarStrBuffer = encoding.GetBytes(grammarStr);
			MemoryStream grammarStream = new MemoryStream(grammarStrBuffer);

			ANTLRInputStream inputStream = new ANTLRInputStream(grammarStream, Encoding.Unicode);
			while (inputStream.Index() < inputStream.Count)
			{
				Console.Out.Write((char)inputStream.LA(1));
				inputStream.Consume();
			}
			Assert.AreEqual(inputStream.Index(), inputStream.Count);
		}

		[Test]
		public void TestConsumeOnANTLRInputStream()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] buffer = encoding.GetBytes("One\r\nTwo");
			MemoryStream grammarStream = new MemoryStream(buffer);

			ANTLRInputStream inputStream = new ANTLRInputStream(grammarStream, Encoding.Unicode);
			Assert.AreEqual(0, inputStream.Index());
			Assert.AreEqual(0, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);

			inputStream.Consume();		// O
			Assert.AreEqual(1, inputStream.Index());
			Assert.AreEqual(1, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);

			inputStream.Consume();		// n
			Assert.AreEqual(2, inputStream.Index());
			Assert.AreEqual(2, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);

			inputStream.Consume();		// e
			Assert.AreEqual(3, inputStream.Index());
			Assert.AreEqual(3, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);

			inputStream.Consume();		// \r
			Assert.AreEqual(4, inputStream.Index());
			Assert.AreEqual(4, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);

			inputStream.Consume();		// \n
			Assert.AreEqual(5, inputStream.Index());
			Assert.AreEqual(0, inputStream.CharPositionInLine);
			Assert.AreEqual(2, inputStream.Line);

			inputStream.Consume();		// T
			Assert.AreEqual(6, inputStream.Index());
			Assert.AreEqual(1, inputStream.CharPositionInLine);
			Assert.AreEqual(2, inputStream.Line);

			inputStream.Consume();		// w
			Assert.AreEqual(7, inputStream.Index());
			Assert.AreEqual(2, inputStream.CharPositionInLine);
			Assert.AreEqual(2, inputStream.Line);

			inputStream.Consume();		// o
			Assert.AreEqual(8, inputStream.Index());
			Assert.AreEqual(3, inputStream.CharPositionInLine);
			Assert.AreEqual(2, inputStream.Line);

			inputStream.Consume();		// EOF
			Assert.AreEqual(8, inputStream.Index());
			Assert.AreEqual(3, inputStream.CharPositionInLine);
			Assert.AreEqual(2, inputStream.Line);

			inputStream.Consume();		// EOF
			Assert.AreEqual(8, inputStream.Index());
			Assert.AreEqual(3, inputStream.CharPositionInLine);
			Assert.AreEqual(2, inputStream.Line);
		}

		[Test]
		public void TestResetOnANTLRInputStream()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] buffer = encoding.GetBytes("One\r\nTwo");
			MemoryStream grammarStream = new MemoryStream(buffer);

			ANTLRInputStream inputStream = new ANTLRInputStream(grammarStream, encoding);
			Assert.AreEqual(0, inputStream.Index());
			Assert.AreEqual(0, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);

			inputStream.Consume();		// O
			inputStream.Consume();		// n

			Assert.AreEqual('e', inputStream.LA(1));
			Assert.AreEqual(2, inputStream.Index());

			inputStream.Reset();
			Assert.AreEqual('O', inputStream.LA(1));
			Assert.AreEqual(0, inputStream.Index());
			Assert.AreEqual(0, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);
			inputStream.Consume();		// O

			Assert.AreEqual('n', inputStream.LA(1));
			Assert.AreEqual(1, inputStream.Index());
			Assert.AreEqual(1, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);
			inputStream.Consume();		// n

			Assert.AreEqual('e', inputStream.LA(1));
			Assert.AreEqual(2, inputStream.Index());
			Assert.AreEqual(2, inputStream.CharPositionInLine);
			Assert.AreEqual(1, inputStream.Line);
			inputStream.Consume();		// e
		}

		[Test]
		public void TestSubstringOnANTLRInputStream()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] buffer = encoding.GetBytes("One\r\nTwo\r\nThree");
			MemoryStream grammarStream = new MemoryStream(buffer);

			ANTLRInputStream stream = new ANTLRInputStream(grammarStream, encoding);

			Assert.AreEqual("Two", stream.Substring(5, 7));
			Assert.AreEqual("One", stream.Substring(0, 2));
			Assert.AreEqual("Three", stream.Substring(10, 14));

			stream.Consume();

			Assert.AreEqual("Two", stream.Substring(5, 7));
			Assert.AreEqual("One", stream.Substring(0, 2));
			Assert.AreEqual("Three", stream.Substring(10, 14));
		}

		[Test]
		public void TestSeekOnANTLRInputStream()
		{
			Encoding encoding = Encoding.Unicode;
			byte[] buffer = encoding.GetBytes("One\r\nTwo\r\nThree");
			MemoryStream grammarStream = new MemoryStream(buffer);

			ANTLRInputStream stream = new ANTLRInputStream(grammarStream, encoding);
			Assert.AreEqual('O', stream.LA(1));
			Assert.AreEqual(0, stream.Index());
			Assert.AreEqual(0, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);

			stream.Seek(6);
			Assert.AreEqual('w', stream.LA(1));
			Assert.AreEqual(6, stream.Index());
			Assert.AreEqual(1, stream.CharPositionInLine);
			Assert.AreEqual(2, stream.Line);

			stream.Seek(11);
			Assert.AreEqual('h', stream.LA(1));
			Assert.AreEqual(11, stream.Index());
			Assert.AreEqual(1, stream.CharPositionInLine);
			Assert.AreEqual(3, stream.Line);

			// seeking backwards leaves state info (other than index in stream) unchanged
			stream.Seek(1);
			Assert.AreEqual('n', stream.LA(1));
			Assert.AreEqual(1, stream.Index());
			Assert.AreEqual(1, stream.CharPositionInLine);
			Assert.AreEqual(3, stream.Line);
		}

		#endregion


		#region ANTLRStringStream Tests

		[Test]
		public void TestSizeOnEmptyANTLRStringStream()
		{
			ANTLRStringStream s1 = new ANTLRStringStream("");
			Assert.AreEqual(0, s1.Count);
			Assert.AreEqual(0, s1.Index());
		}

		[Test]
		public void TestSizeOnANTLRStringStream()
		{
			ANTLRStringStream s1 = new ANTLRStringStream("lexer\r\n");
			Assert.AreEqual(7, s1.Count);

			ANTLRStringStream s2 = new ANTLRStringStream(grammarStr);
			Assert.AreEqual(grammarStr.Length, s2.Count);

			ANTLRStringStream s3 = new ANTLRStringStream("grammar P;");
			Assert.AreEqual(10, s3.Count);
		}

		[Test]
		public void TestConsumeOnANTLRStringStream()
		{
			ANTLRStringStream stream = new ANTLRStringStream("One\r\nTwo");
			Assert.AreEqual(0, stream.Index());
			Assert.AreEqual(0, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);

			stream.Consume();		// O
			Assert.AreEqual(1, stream.Index());
			Assert.AreEqual(1, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);

			stream.Consume();		// n
			Assert.AreEqual(2, stream.Index());
			Assert.AreEqual(2, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);

			stream.Consume();		// e
			Assert.AreEqual(3, stream.Index());
			Assert.AreEqual(3, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);

			stream.Consume();		// \r
			Assert.AreEqual(4, stream.Index());
			Assert.AreEqual(4, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);

			stream.Consume();		// \n
			Assert.AreEqual(5, stream.Index());
			Assert.AreEqual(0, stream.CharPositionInLine);
			Assert.AreEqual(2, stream.Line);

			stream.Consume();		// T
			Assert.AreEqual(6, stream.Index());
			Assert.AreEqual(1, stream.CharPositionInLine);
			Assert.AreEqual(2, stream.Line);

			stream.Consume();		// w
			Assert.AreEqual(7, stream.Index());
			Assert.AreEqual(2, stream.CharPositionInLine);
			Assert.AreEqual(2, stream.Line);

			stream.Consume();		// o
			Assert.AreEqual(8, stream.Index());
			Assert.AreEqual(3, stream.CharPositionInLine);
			Assert.AreEqual(2, stream.Line);

			stream.Consume();		// EOF
			Assert.AreEqual(8, stream.Index());
			Assert.AreEqual(3, stream.CharPositionInLine);
			Assert.AreEqual(2, stream.Line);

			stream.Consume();		// EOF
			Assert.AreEqual(8, stream.Index());
			Assert.AreEqual(3, stream.CharPositionInLine);
			Assert.AreEqual(2, stream.Line);
		}

		[Test]
		public void TestResetOnANTLRStringStream()
		{
			ANTLRStringStream stream = new ANTLRStringStream("One\r\nTwo");
			Assert.AreEqual(0, stream.Index());
			Assert.AreEqual(0, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);

			stream.Consume();		// O
			stream.Consume();		// n

			Assert.AreEqual('e', stream.LA(1));
			Assert.AreEqual(2, stream.Index());

			stream.Reset();
			Assert.AreEqual('O', stream.LA(1));
			Assert.AreEqual(0, stream.Index());
			Assert.AreEqual(0, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);
			stream.Consume();		// O

			Assert.AreEqual('n', stream.LA(1));
			Assert.AreEqual(1, stream.Index());
			Assert.AreEqual(1, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);
			stream.Consume();		// n

			Assert.AreEqual('e', stream.LA(1));
			Assert.AreEqual(2, stream.Index());
			Assert.AreEqual(2, stream.CharPositionInLine);
			Assert.AreEqual(1, stream.Line);
			stream.Consume();		// e
		}

		[Test]
		public void TestSubstringOnANTLRStringStream()
		{
			ANTLRStringStream stream = new ANTLRStringStream("One\r\nTwo\r\nThree");

			Assert.AreEqual("Two", stream.Substring(5, 7));
			Assert.AreEqual("One", stream.Substring(0, 2));
			Assert.AreEqual("Three", stream.Substring(10, 14));

			stream.Consume();

			Assert.AreEqual("Two", stream.Substring(5, 7));
			Assert.AreEqual("One", stream.Substring(0, 2));
			Assert.AreEqual("Three", stream.Substring(10, 14));
		}

		#endregion
	}
}