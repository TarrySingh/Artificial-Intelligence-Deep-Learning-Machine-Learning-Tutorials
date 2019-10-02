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

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestSyntacticPredicateEvaluation extends BaseTest {
	@Test public void testTwoPredsWithNakedAlt() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : (a ';')+ ;\n" +
			"a\n" +
			"options {\n" +
			"  k=1;\n" +
			"}\n" +
			"  : (b '.')=> b '.' {System.out.println(\"alt 1\");}\n" +
			"  | (b)=> b {System.out.println(\"alt 2\");}\n" +
			"  | c       {System.out.println(\"alt 3\");}\n" +
			"  ;\n" +
			"b\n" +
			"@init {System.out.println(\"enter b\");}\n" +
			"   : '(' 'x' ')' ;\n" +
			"c\n" +
			"@init {System.out.println(\"enter c\");}\n" +
			"   : '(' c ')' | 'x' ;\n" +
			"WS : (' '|'\\n')+ {$channel=HIDDEN;}\n" +
			"   ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "(x) ;", false);
		String expecting =
			"enter b\n" +
			"enter b\n" +
			"enter b\n" +
			"alt 2\n";
		assertEquals(expecting, found);

		found = execParser("T.g", grammar, "TParser", "TLexer",
			    "a", "(x). ;", false);
		expecting =
			"enter b\n" +
			"enter b\n" +
			"alt 1\n";
		assertEquals(expecting, found);

		found = execParser("T.g", grammar, "TParser", "TLexer",
			    "a", "((x)) ;", false);
		expecting =
			"enter b\n" +
			"enter b\n" +
			"enter c\n" +
			"enter c\n" +
			"enter c\n" +
			"alt 3\n";
		assertEquals(expecting, found);
	}

	@Test public void testTwoPredsWithNakedAltNotLast() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : (a ';')+ ;\n" +
			"a\n" +
			"options {\n" +
			"  k=1;\n" +
			"}\n" +
			"  : (b '.')=> b '.' {System.out.println(\"alt 1\");}\n" +
			"  | c       {System.out.println(\"alt 2\");}\n" +
			"  | (b)=> b {System.out.println(\"alt 3\");}\n" +
			"  ;\n" +
			"b\n" +
			"@init {System.out.println(\"enter b\");}\n" +
			"   : '(' 'x' ')' ;\n" +
			"c\n" +
			"@init {System.out.println(\"enter c\");}\n" +
			"   : '(' c ')' | 'x' ;\n" +
			"WS : (' '|'\\n')+ {$channel=HIDDEN;}\n" +
			"   ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "(x) ;", false);
		String expecting =
			"enter b\n" +
			"enter c\n" +
			"enter c\n" +
			"alt 2\n";
		assertEquals(expecting, found);

		found = execParser("T.g", grammar, "TParser", "TLexer",
			    "a", "(x). ;", false);
		expecting =
			"enter b\n" +
			"enter b\n" +
			"alt 1\n";
		assertEquals(expecting, found);

		found = execParser("T.g", grammar, "TParser", "TLexer",
			    "a", "((x)) ;", false);
		expecting =
			"enter b\n" +
			"enter c\n" +
			"enter c\n" +
			"enter c\n" +
			"alt 2\n";
		assertEquals(expecting, found);
	}

	@Test public void testLexerPred() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : A ;\n" +
			"A options {k=1;}\n" + // force backtracking
			"  : (B '.')=>B '.' {System.out.println(\"alt1\");}\n" +
			"  | B {System.out.println(\"alt2\");}" +
			"  ;\n" +
			"fragment\n" +
			"B : 'x'+ ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "s", "xxx", false);

		assertEquals("alt2\n", found);

		found = execParser("T.g", grammar, "TParser", "TLexer",
			    "s", "xxx.", false);

		assertEquals("alt1\n", found);
	}

	@Test public void testLexerWithPredLongerThanAlt() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : A ;\n" +
			"A options {k=1;}\n" + // force backtracking
			"  : (B '.')=>B {System.out.println(\"alt1\");}\n" +
			"  | B {System.out.println(\"alt2\");}" +
			"  ;\n" +
			"D : '.' {System.out.println(\"D\");} ;\n" +
			"fragment\n" +
			"B : 'x'+ ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "s", "xxx", false);

		assertEquals("alt2\n", found);

		found = execParser("T.g", grammar, "TParser", "TLexer",
			    "s", "xxx.", false);

		assertEquals("alt1\nD\n", found);
	}

	@Test public void testLexerPredCyclicPrediction() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : A ;\n" +
			"A : (B)=>(B|'y'+) {System.out.println(\"alt1\");}\n" +
			"  | B {System.out.println(\"alt2\");}\n" +
			"  | 'y'+ ';'" +
			"  ;\n" +
			"fragment\n" +
			"B : 'x'+ ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "s", "xxx", false);

		assertEquals("alt1\n", found);
	}

	@Test public void testLexerPredCyclicPrediction2() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : A ;\n" +
			"A : (B '.')=>(B|'y'+) {System.out.println(\"alt1\");}\n" +
			"  | B {System.out.println(\"alt2\");}\n" +
			"  | 'y'+ ';'" +
			"  ;\n" +
			"fragment\n" +
			"B : 'x'+ ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "s", "xxx", false);
		assertEquals("alt2\n", found);
	}

	@Test public void testSimpleNestedPred() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : (expr ';')+ ;\n" +
			"expr\n" +
			"options {\n" +
			"  k=1;\n" +
			"}\n" +
			"@init {System.out.println(\"enter expr \"+input.LT(1).getText());}\n" +
			"  : (atom 'x') => atom 'x'\n" +
			"  | atom\n" +
			";\n" +
			"atom\n" +
			"@init {System.out.println(\"enter atom \"+input.LT(1).getText());}\n" +
			"   : '(' expr ')'\n" +
			"   | INT\n" +
			"   ;\n" +
			"INT: '0'..'9'+ ;\n" +
			"WS : (' '|'\\n')+ {$channel=HIDDEN;}\n" +
			"   ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "s", "(34)x;", false);
		String expecting =
			"enter expr (\n" +
			"enter atom (\n" +
			"enter expr 34\n" +
			"enter atom 34\n" +
			"enter atom 34\n" +
			"enter atom (\n" +
			"enter expr 34\n" +
			"enter atom 34\n" +
			"enter atom 34\n";
		assertEquals(expecting, found);
	}

	@Test public void testTripleNestedPredInLexer() throws Exception {
		String grammar =
			"grammar T;\n" +
			"s : (.)+ {System.out.println(\"done\");} ;\n" +
			"EXPR\n" +
			"options {\n" +
			"  k=1;\n" +
			"}\n" +
			"@init {System.out.println(\"enter expr \"+(char)input.LT(1));}\n" +
			"  : (ATOM 'x') => ATOM 'x' {System.out.println(\"ATOM x\");}\n" +
			"  | ATOM {System.out.println(\"ATOM \"+$ATOM.text);}\n" +
			";\n" +
			"fragment ATOM\n" +
			"@init {System.out.println(\"enter atom \"+(char)input.LT(1));}\n" +
			"   : '(' EXPR ')'\n" +
			"   | INT\n" +
			"   ;\n" +
			"fragment INT: '0'..'9'+ ;\n" +
			"fragment WS : (' '|'\\n')+ \n" +
			"   ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "s", "((34)x)x", false);
		String expecting = // has no memoization
			"enter expr (\n" +
			"enter atom (\n" +
			"enter expr (\n" +
			"enter atom (\n" +
			"enter expr 3\n" +
			"enter atom 3\n" +
			"enter atom 3\n" +
			"enter atom (\n" +
			"enter expr 3\n" +
			"enter atom 3\n" +
			"enter atom 3\n" +
			"enter atom (\n" +
			"enter expr (\n" +
			"enter atom (\n" +
			"enter expr 3\n" +
			"enter atom 3\n" +
			"enter atom 3\n" +
			"enter atom (\n" +
			"enter expr 3\n" +
			"enter atom 3\n" +
			"enter atom 3\n" +
			"ATOM 34\n" +
			"ATOM x\n" +
			"ATOM x\n" +
			"done\n";
		assertEquals(expecting, found);
	}

	@Test public void testTreeParserWithSynPred() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT+ (PERIOD|SEMI);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"SEMI : ';' ;\n"+
			"PERIOD : '.' ;\n"+
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n" +
			"options {k=1; backtrack=true; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ID INT+ PERIOD {System.out.print(\"alt 1\");}"+
			"  | ID INT+ SEMI   {System.out.print(\"alt 2\");}\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
				    treeGrammar, "TP", "TLexer", "a", "a", "a 1 2 3;");
		assertEquals("alt 2\n", found);
	}

	@Test public void testTreeParserWithNestedSynPred() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT+ (PERIOD|SEMI);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"SEMI : ';' ;\n"+
			"PERIOD : '.' ;\n"+
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		// backtracks in a and b due to k=1
		String treeGrammar =
			"tree grammar TP;\n" +
			"options {k=1; backtrack=true; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ID b {System.out.print(\" a:alt 1\");}"+
			"  | ID INT+ SEMI   {System.out.print(\" a:alt 2\");}\n" +
			"  ;\n" +
			"b : INT PERIOD  {System.out.print(\"b:alt 1\");}" + // choose this alt for just one INT
			"  | INT+ PERIOD {System.out.print(\"b:alt 2\");}" +
			"  ;";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
				    treeGrammar, "TP", "TLexer", "a", "a", "a 1 2 3.");
		assertEquals("b:alt 2 a:alt 1\n", found);
	}

	@Test public void testSynPredWithOutputTemplate() throws Exception {
		// really just seeing if it will compile
		String grammar =
			"grammar T;\n" +
			"options {output=template;}\n" +
			"a\n" +
			"options {\n" +
			"  k=1;\n" +
			"}\n" +
			"  : ('x'+ 'y')=> 'x'+ 'y' -> template(a={$text}) <<1:<a>;>>\n" +
			"  | 'x'+ 'z' -> template(a={$text}) <<2:<a>;>>\n"+
			"  ;\n" +
			"WS : (' '|'\\n')+ {$channel=HIDDEN;}\n" +
			"   ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "xxxy", false);

		assertEquals("1:xxxy;\n", found);
	}

	@Test public void testSynPredWithOutputAST() throws Exception {
		// really just seeing if it will compile
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a\n" +
			"options {\n" +
			"  k=1;\n" +
			"}\n" +
			"  : ('x'+ 'y')=> 'x'+ 'y'\n" +
			"  | 'x'+ 'z'\n"+
			"  ;\n" +
			"WS : (' '|'\\n')+ {$channel=HIDDEN;}\n" +
			"   ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "xxxy", false);

		assertEquals("x x x y\n", found);
	}

	@Test public void testOptionalBlockWithSynPred() throws Exception {
		String grammar =
			"grammar T;\n" +
				"\n" +
				"a : ( (b)=> b {System.out.println(\"b\");})? b ;\n" +
				"b : 'x' ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "xx", false);
		assertEquals("b\n", found);
		found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "x", false);
		assertEquals("", found);
	}

	@Test public void testSynPredK2() throws Exception {
		// all manually specified syn predicates are gated (i.e., forced
		// to execute).
		String grammar =
			"grammar T;\n" +
				"\n" +
				"a : (b)=> b {System.out.println(\"alt1\");} | 'a' 'c' ;\n" +
				"b : 'a' 'b' ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "ab", false);

		assertEquals("alt1\n", found);
	}

	@Test public void testSynPredKStar() throws Exception {
		String grammar =
			"grammar T;\n" +
				"\n" +
				"a : (b)=> b {System.out.println(\"alt1\");} | 'a'+ 'c' ;\n" +
				"b : 'a'+ 'b' ;\n" ;
		String found = execParser("T.g", grammar, "TParser", "TLexer",
				    "a", "aaab", false);

		assertEquals("alt1\n", found);
	}

}
