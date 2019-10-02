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

import org.antlr.tool.Grammar;
import org.antlr.tool.Interpreter;
import org.antlr.runtime.*;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestInterpretedLexing extends BaseTest {

	/*
	static class Tracer implements ANTLRDebugInterface {
		Grammar g;
		public DebugActions(Grammar g) {
			this.g = g;
		}
		public void enterRule(String ruleName) {
			System.out.println("enterRule("+ruleName+")");
		}

		public void exitRule(String ruleName) {
			System.out.println("exitRule("+ruleName+")");
		}

		public void matchElement(int type) {
			System.out.println("matchElement("+g.getTokenName(type)+")");
		}

		public void mismatchedElement(MismatchedTokenException e) {
			System.out.println(e);
			e.printStackTrace(System.out);
		}

		public void mismatchedSet(MismatchedSetException e) {
			System.out.println(e);
			e.printStackTrace(System.out);
		}

		public void noViableAlt(NoViableAltException e) {
			System.out.println(e);
			e.printStackTrace(System.out);
		}
	}
    */

    /** Public default constructor used by TestRig */
    public TestInterpretedLexing() {
    }

	@Test public void testSimpleAltCharTest() throws Exception {
        Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "A : 'a' | 'b' | 'c';");
		final int Atype = g.getTokenType("A");
        Interpreter engine = new Interpreter(g, new ANTLRStringStream("a"));
        engine = new Interpreter(g, new ANTLRStringStream("b"));
		Token result = engine.scan("A");
		assertEquals(result.getType(), Atype);
        engine = new Interpreter(g, new ANTLRStringStream("c"));
		result = engine.scan("A");
		assertEquals(result.getType(), Atype);
    }

    @Test public void testSingleRuleRef() throws Exception {
        Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "A : 'a' B 'c' ;\n" +
                "B : 'b' ;\n");
		final int Atype = g.getTokenType("A");
		Interpreter engine = new Interpreter(g, new ANTLRStringStream("abc")); // should ignore the x
		Token result = engine.scan("A");
		assertEquals(result.getType(), Atype);
    }

    @Test public void testSimpleLoop() throws Exception {
        Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "INT : (DIGIT)+ ;\n"+
				"fragment DIGIT : '0'..'9';\n");
		final int INTtype = g.getTokenType("INT");
		Interpreter engine = new Interpreter(g, new ANTLRStringStream("12x")); // should ignore the x
		Token result = engine.scan("INT");
		assertEquals(result.getType(), INTtype);
		engine = new Interpreter(g, new ANTLRStringStream("1234"));
		result = engine.scan("INT");
		assertEquals(result.getType(), INTtype);
    }

    @Test public void testMultAltLoop() throws Exception {
		Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "A : ('0'..'9'|'a'|'b')+ ;\n");
		final int Atype = g.getTokenType("A");
		Interpreter engine = new Interpreter(g, new ANTLRStringStream("a"));
		Token result = engine.scan("A");
        engine = new Interpreter(g, new ANTLRStringStream("a"));
		result = engine.scan("A");
		assertEquals(result.getType(), Atype);
		engine = new Interpreter(g, new ANTLRStringStream("1234"));
		result = engine.scan("A");
		assertEquals(result.getType(), Atype);
        engine = new Interpreter(g, new ANTLRStringStream("aaa"));
		result = engine.scan("A");
		assertEquals(result.getType(), Atype);
        engine = new Interpreter(g, new ANTLRStringStream("aaaa9"));
		result = engine.scan("A");
		assertEquals(result.getType(), Atype);
        engine = new Interpreter(g, new ANTLRStringStream("b"));
		result = engine.scan("A");
		assertEquals(result.getType(), Atype);
        engine = new Interpreter(g, new ANTLRStringStream("baa"));
		result = engine.scan("A");
		assertEquals(result.getType(), Atype);
    }

	@Test public void testSimpleLoops() throws Exception {
		Grammar g = new Grammar(
				"lexer grammar t;\n"+
				"A : ('0'..'9')+ '.' ('0'..'9')* | ('0'..'9')+ ;\n");
		final int Atype = g.getTokenType("A");
		CharStream input = new ANTLRStringStream("1234.5");
		Interpreter engine = new Interpreter(g, input);
		Token result = engine.scan("A");
		assertEquals(result.getType(), Atype);
	}

	@Test public void testTokensRules() throws Exception {
		Grammar pg = new Grammar(
			"grammar p;\n"+
			"a : (INT|FLOAT|WS)+;\n");
		Grammar g = new Grammar();
		g.importTokenVocabulary(pg);
		g.setFileName("<string>");
		g.setGrammarContent(
			"lexer grammar t;\n"+
			"INT : (DIGIT)+ ;\n"+
			"FLOAT : (DIGIT)+ '.' (DIGIT)* ;\n"+
			"fragment DIGIT : '0'..'9';\n" +
			"WS : (' ')+ {channel=99;};\n");
		CharStream input = new ANTLRStringStream("123 139.52");
		Interpreter lexEngine = new Interpreter(g, input);

		CommonTokenStream tokens = new CommonTokenStream(lexEngine);
		String result = tokens.toString();
		//System.out.println(result);
		String expecting = "123 139.52";
		assertEquals(result,expecting);
	}

}
