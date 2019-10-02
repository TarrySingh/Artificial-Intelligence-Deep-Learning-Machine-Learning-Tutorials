/*
 [The "BSD licence"]
 Copyright (c) 2005-2006 Terence Parr
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

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
package org.antlr.test;

import org.antlr.runtime.ANTLRStringStream;
import org.antlr.runtime.CharStream;
import org.antlr.runtime.TokenRewriteStream;
import org.antlr.tool.Grammar;
import org.antlr.tool.Interpreter;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestTokenRewriteStream extends BaseTest {

    /** Public default constructor used by TestRig */
    public TestTokenRewriteStream() {
    }

	@Test public void testInsertBeforeIndex0() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(0, "0");
		String result = tokens.toString();
		String expecting = "0abc";
		assertEquals(expecting, result);
	}

	@Test public void testInsertAfterLastIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertAfter(2, "x");
		String result = tokens.toString();
		String expecting = "abcx";
		assertEquals(expecting, result);
	}

	@Test public void test2InsertBeforeAfterMiddleIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(1, "x");
		tokens.insertAfter(1, "x");
		String result = tokens.toString();
		String expecting = "axbxc";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceIndex0() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(0, "x");
		String result = tokens.toString();
		String expecting = "xbc";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceLastIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, "x");
		String result = tokens.toString();
		String expecting = "abx";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceMiddleIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(1, "x");
		String result = tokens.toString();
		String expecting = "axc";
		assertEquals(expecting, result);
	}

    @Test public void testToStringStartStop() throws Exception {
        Grammar g = new Grammar(
            "lexer grammar t;\n"+
            "ID : 'a'..'z'+;\n" +
            "INT : '0'..'9'+;\n" +
            "SEMI : ';';\n" +
            "MUL : '*';\n" +
            "ASSIGN : '=';\n" +
            "WS : ' '+;\n");
        // Tokens: 0123456789
        // Input:  x = 3 * 0;
        CharStream input = new ANTLRStringStream("x = 3 * 0;");
        Interpreter lexEngine = new Interpreter(g, input);
        TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
        tokens.LT(1); // fill buffer
        tokens.replace(4, 8, "0"); // replace 3 * 0 with 0

        String result = tokens.toOriginalString();
        String expecting = "x = 3 * 0;";
        assertEquals(expecting, result);

        result = tokens.toString();
        expecting = "x = 0;";
        assertEquals(expecting, result);

        result = tokens.toString(0,9);
        expecting = "x = 0;";
        assertEquals(expecting, result);

        result = tokens.toString(4,8);
        expecting = "0";
        assertEquals(expecting, result);
    }

    @Test public void testToStringStartStop2() throws Exception {
        Grammar g = new Grammar(
            "lexer grammar t;\n"+
            "ID : 'a'..'z'+;\n" +
            "INT : '0'..'9'+;\n" +
            "SEMI : ';';\n" +
            "ASSIGN : '=';\n" +
            "PLUS : '+';\n" +
            "MULT : '*';\n" +
            "WS : ' '+;\n");
        // Tokens: 012345678901234567
        // Input:  x = 3 * 0 + 2 * 0;
        CharStream input = new ANTLRStringStream("x = 3 * 0 + 2 * 0;");
        Interpreter lexEngine = new Interpreter(g, input);
        TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
        tokens.LT(1); // fill buffer

        String result = tokens.toOriginalString();
        String expecting = "x = 3 * 0 + 2 * 0;";
        assertEquals(expecting, result);

        tokens.replace(4, 8, "0"); // replace 3 * 0 with 0
        result = tokens.toString();
        expecting = "x = 0 + 2 * 0;";
        assertEquals(expecting, result);

        result = tokens.toString(0,17);
        expecting = "x = 0 + 2 * 0;";
        assertEquals(expecting, result);

        result = tokens.toString(4,8);
        expecting = "0";
        assertEquals(expecting, result);

        result = tokens.toString(0,8);
        expecting = "x = 0";
        assertEquals(expecting, result);

        result = tokens.toString(12,16);
        expecting = "2 * 0";
        assertEquals(expecting, result);

        tokens.insertAfter(17, "// comment");
        result = tokens.toString(12,17);
        expecting = "2 * 0;// comment";
        assertEquals(expecting, result);

        result = tokens.toString(0,8); // try again after insert at end
        expecting = "x = 0";
        assertEquals(expecting, result);
    }


    @Test public void test2ReplaceMiddleIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(1, "x");
		tokens.replace(1, "y");
		String result = tokens.toString();
		String expecting = "ayc";
		assertEquals(expecting, result);
	}

    @Test public void test2ReplaceMiddleIndex1InsertBefore() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
        tokens.insertBefore(0, "_");
        tokens.replace(1, "x");
		tokens.replace(1, "y");
		String result = tokens.toString();
		String expecting = "_ayc";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceThenDeleteMiddleIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(1, "x");
		tokens.delete(1);
		String result = tokens.toString();
		String expecting = "ac";
		assertEquals(expecting, result);
	}

	@Test public void testInsertInPriorReplace() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(0, 2, "x");
		tokens.insertBefore(1, "0");
		Exception exc = null;
		try {
			tokens.toString();
		}
		catch (IllegalArgumentException iae) {
			exc = iae;
		}
		String expecting = "insert op <InsertBeforeOp@1:\"0\"> within boundaries of previous <ReplaceOp@0..2:\"x\">";
		assertNotNull(exc);
		assertEquals(expecting, exc.getMessage());
	}

	@Test public void testInsertThenReplaceSameIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(0, "0");
		tokens.replace(0, "x"); // supercedes insert at 0
		String result = tokens.toString();
		String expecting = "xbc";
		assertEquals(expecting, result);
	}

	@Test public void test2InsertMiddleIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(1, "x");
		tokens.insertBefore(1, "y");
		String result = tokens.toString();
		String expecting = "ayxbc";
		assertEquals(expecting, result);
	}

	@Test public void test2InsertThenReplaceIndex0() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(0, "x");
		tokens.insertBefore(0, "y");
		tokens.replace(0, "z");
		String result = tokens.toString();
		String expecting = "zbc";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceThenInsertBeforeLastIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, "x");
		tokens.insertBefore(2, "y");
		String result = tokens.toString();
		String expecting = "abyx";
		assertEquals(expecting, result);
	}

	@Test public void testInsertThenReplaceLastIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(2, "y");
		tokens.replace(2, "x");
		String result = tokens.toString();
		String expecting = "abx";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceThenInsertAfterLastIndex() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, "x");
		tokens.insertAfter(2, "y");
		String result = tokens.toString();
		String expecting = "abxy";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceRangeThenInsertAtLeftEdge() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcccba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 4, "x");
		tokens.insertBefore(2, "y");
		String result = tokens.toString();
		String expecting = "abyxba";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceRangeThenInsertAtRightEdge() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcccba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 4, "x");
		tokens.insertBefore(4, "y"); // no effect; within range of a replace
		Exception exc = null;
		try {
			tokens.toString();
		}
		catch (IllegalArgumentException iae) {
			exc = iae;
		}
		String expecting = "insert op <InsertBeforeOp@4:\"y\"> within boundaries of previous <ReplaceOp@2..4:\"x\">";
		assertNotNull(exc);
		assertEquals(expecting, exc.getMessage());
	}

	@Test public void testReplaceRangeThenInsertAfterRightEdge() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcccba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 4, "x");
		tokens.insertAfter(4, "y");
		String result = tokens.toString();
		String expecting = "abxyba";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceAll() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcccba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(0, 6, "x");
		String result = tokens.toString();
		String expecting = "x";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceSubsetThenFetch() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcccba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 4, "xyz");
		String result = tokens.toString(0,6);
		String expecting = "abxyzba";
		assertEquals(expecting, result);
	}

	@Test public void testReplaceThenReplaceSuperset() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcccba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 4, "xyz");
		tokens.replace(3, 5, "foo"); // overlaps, error
		Exception exc = null;
		try {
			tokens.toString();
		}
		catch (IllegalArgumentException iae) {
			exc = iae;
		}
		String expecting = "replace op boundaries of <ReplaceOp@3..5:\"foo\"> overlap with previous <ReplaceOp@2..4:\"xyz\">";
		assertNotNull(exc);
		assertEquals(expecting, exc.getMessage());
	}

	@Test public void testReplaceThenReplaceLowerIndexedSuperset() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcccba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 4, "xyz");
		tokens.replace(1, 3, "foo"); // overlap, error
		Exception exc = null;
		try {
			tokens.toString();
		}
		catch (IllegalArgumentException iae) {
			exc = iae;
		}
		String expecting = "replace op boundaries of <ReplaceOp@1..3:\"foo\"> overlap with previous <ReplaceOp@2..4:\"xyz\">";
		assertNotNull(exc);
		assertEquals(expecting, exc.getMessage());
	}

	@Test public void testReplaceSingleMiddleThenOverlappingSuperset() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcba");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 2, "xyz");
		tokens.replace(0, 3, "foo");
		String result = tokens.toString();
		String expecting = "fooa";
		assertEquals(expecting, result);
	}

	// June 2, 2008 I rewrote core of rewrite engine; just adding lots more tests here

	@Test public void testCombineInserts() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(0, "x");
		tokens.insertBefore(0, "y");
		String result = tokens.toString();
		String expecting = "yxabc";
		assertEquals(expecting, result);
	}

	@Test public void testCombine3Inserts() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(1, "x");
		tokens.insertBefore(0, "y");
		tokens.insertBefore(1, "z");
		String result = tokens.toString();
		String expecting = "yazxbc";
		assertEquals(expecting, result);
	}

	@Test public void testCombineInsertOnLeftWithReplace() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(0, 2, "foo");
		tokens.insertBefore(0, "z"); // combine with left edge of rewrite
		String result = tokens.toString();
		String expecting = "zfoo";
		assertEquals(expecting, result);
	}

	@Test public void testCombineInsertOnLeftWithDelete() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.delete(0, 2);
		tokens.insertBefore(0, "z"); // combine with left edge of rewrite
		String result = tokens.toString();
		String expecting = "z"; // make sure combo is not znull
		assertEquals(expecting, result);
	}

	@Test public void testDisjointInserts() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(1, "x");
		tokens.insertBefore(2, "y");
		tokens.insertBefore(0, "z");
		String result = tokens.toString();
		String expecting = "zaxbyc";
		assertEquals(expecting, result);
	}

	@Test public void testOverlappingReplace() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(1, 2, "foo");
		tokens.replace(0, 3, "bar"); // wipes prior nested replace
		String result = tokens.toString();
		String expecting = "bar";
		assertEquals(expecting, result);
	}

	@Test public void testOverlappingReplace2() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(0, 3, "bar");
		tokens.replace(1, 2, "foo"); // cannot split earlier replace
		Exception exc = null;
		try {
			tokens.toString();
		}
		catch (IllegalArgumentException iae) {
			exc = iae;
		}
		String expecting = "replace op boundaries of <ReplaceOp@1..2:\"foo\"> overlap with previous <ReplaceOp@0..3:\"bar\">";
		assertNotNull(exc);
		assertEquals(expecting, exc.getMessage());
	}

	@Test public void testOverlappingReplace3() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(1, 2, "foo");
		tokens.replace(0, 2, "bar"); // wipes prior nested replace
		String result = tokens.toString();
		String expecting = "barc";
		assertEquals(expecting, result);
	}

	@Test public void testOverlappingReplace4() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(1, 2, "foo");
		tokens.replace(1, 3, "bar"); // wipes prior nested replace
		String result = tokens.toString();
		String expecting = "abar";
		assertEquals(expecting, result);
	}

	@Test public void testDropIdenticalReplace() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(1, 2, "foo");
		tokens.replace(1, 2, "foo"); // drop previous, identical
		String result = tokens.toString();
		String expecting = "afooc";
		assertEquals(expecting, result);
	}

	@Test public void testDropPrevCoveredInsert() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(1, "foo");
		tokens.replace(1, 2, "foo"); // kill prev insert
		String result = tokens.toString();
		String expecting = "afooc";
		assertEquals(expecting, result);
	}

	@Test public void testLeaveAloneDisjointInsert() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.insertBefore(1, "x");
		tokens.replace(2, 3, "foo");
		String result = tokens.toString();
		String expecting = "axbfoo";
		assertEquals(expecting, result);
	}

	@Test public void testLeaveAloneDisjointInsert2() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';\n" +
			"B : 'b';\n" +
			"C : 'c';\n");
		CharStream input = new ANTLRStringStream("abcc");
		Interpreter lexEngine = new Interpreter(g, input);
		TokenRewriteStream tokens = new TokenRewriteStream(lexEngine);
		tokens.LT(1); // fill buffer
		tokens.replace(2, 3, "foo");
		tokens.insertBefore(1, "x");
		String result = tokens.toString();
		String expecting = "axbfoo";
		assertEquals(expecting, result);
	}

}
