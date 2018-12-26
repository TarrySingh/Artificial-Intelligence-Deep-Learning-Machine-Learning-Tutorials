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

public class TestSemanticPredicateEvaluation extends BaseTest {
	@Test public void testSimpleCyclicDFAWithPredicate() throws Exception {
		String grammar =
			"grammar foo;\n" +
			"a : {false}? 'x'* 'y' {System.out.println(\"alt1\");}\n" +
			"  | {true}?  'x'* 'y' {System.out.println(\"alt2\");}\n" +
			"  ;\n" ;
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "xxxy", false);
		assertEquals("alt2\n", found);
	}

	@Test public void testSimpleCyclicDFAWithInstanceVarPredicate() throws Exception {
		String grammar =
			"grammar foo;\n" +
			"@members {boolean v=true;}\n" +
			"a : {false}? 'x'* 'y' {System.out.println(\"alt1\");}\n" +
			"  | {v}?     'x'* 'y' {System.out.println(\"alt2\");}\n" +
			"  ;\n" ;
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "xxxy", false);
		assertEquals("alt2\n", found);
	}

	@Test public void testPredicateValidation() throws Exception {
		String grammar =
			"grammar foo;\n" +
			"@members {\n" +
			"public void reportError(RecognitionException e) {\n" +
			"    System.out.println(\"error: \"+e.toString());\n" +
			"}\n" +
			"}\n" +
			"\n" +
			"a : {false}? 'x'\n" +
			"  ;\n" ;
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "x", false);
		assertEquals("error: FailedPredicateException(a,{false}?)\n", found);
	}

	@Test public void testLexerPreds() throws Exception {
		String grammar =
			"grammar foo;" +
			"@lexer::members {boolean p=false;}\n" +
			"a : (A|B)+ ;\n" +
			"A : {p}? 'a'  {System.out.println(\"token 1\");} ;\n" +
			"B : {!p}? 'a' {System.out.println(\"token 2\");} ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "a", false);
		// "a" is ambig; can match both A, B.  Pred says match 2
		assertEquals("token 2\n", found);
	}

	@Test public void testLexerPreds2() throws Exception {
		String grammar =
			"grammar foo;" +
			"@lexer::members {boolean p=true;}\n" +
			"a : (A|B)+ ;\n" +
			"A : {p}? 'a' {System.out.println(\"token 1\");} ;\n" +
			"B : ('a'|'b')+ {System.out.println(\"token 2\");} ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "a", false);
		// "a" is ambig; can match both A, B.  Pred says match 1
		assertEquals("token 1\n", found);
	}

	@Test public void testLexerPredInExitBranch() throws Exception {
		// p says it's ok to exit; it has precendence over the !p loopback branch
		String grammar =
			"grammar foo;" +
			"@lexer::members {boolean p=true;}\n" +
			"a : (A|B)+ ;\n" +
			"A : ('a' {System.out.print(\"1\");})*\n" +
			"    {p}?\n" +
			"    ('a' {System.out.print(\"2\");})* ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aaa", false);
		assertEquals("222\n", found);
	}

	@Test public void testLexerPredInExitBranch2() throws Exception {
		String grammar =
			"grammar foo;" +
			"@lexer::members {boolean p=true;}\n" +
			"a : (A|B)+ ;\n" +
			"A : ({p}? 'a' {System.out.print(\"1\");})*\n" +
			"    ('a' {System.out.print(\"2\");})* ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aaa", false);
		assertEquals("111\n", found);
	}

	@Test public void testLexerPredInExitBranch3() throws Exception {
		String grammar =
			"grammar foo;" +
			"@lexer::members {boolean p=true;}\n" +
			"a : (A|B)+ ;\n" +
			"A : ({p}? 'a' {System.out.print(\"1\");} | )\n" +
			"    ('a' {System.out.print(\"2\");})* ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aaa", false);
		assertEquals("122\n", found);
	}

	@Test public void testLexerPredInExitBranch4() throws Exception {
		String grammar =
			"grammar foo;" +
			"a : (A|B)+ ;\n" +
			"A @init {int n=0;} : ({n<2}? 'a' {System.out.print(n++);})+\n" +
			"    ('a' {System.out.print(\"x\");})* ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aaaaa", false);
		assertEquals("01xxx\n", found);
	}

	@Test public void testLexerPredsInCyclicDFA() throws Exception {
		String grammar =
			"grammar foo;" +
			"@lexer::members {boolean p=false;}\n" +
			"a : (A|B)+ ;\n" +
			"A : {p}? ('a')+ 'x'  {System.out.println(\"token 1\");} ;\n" +
			"B :      ('a')+ 'x' {System.out.println(\"token 2\");} ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aax", false);
		assertEquals("token 2\n", found);
	}

	@Test public void testLexerPredsInCyclicDFA2() throws Exception {
		String grammar =
			"grammar foo;" +
			"@lexer::members {boolean p=false;}\n" +
			"a : (A|B)+ ;\n" +
			"A : {p}? ('a')+ 'x' ('y')? {System.out.println(\"token 1\");} ;\n" +
			"B :      ('a')+ 'x' {System.out.println(\"token 2\");} ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aax", false);
		assertEquals("token 2\n", found);
	}

	@Test public void testGatedPred() throws Exception {
		String grammar =
			"grammar foo;" +
			"a : (A|B)+ ;\n" +
			"A : {true}?=> 'a' {System.out.println(\"token 1\");} ;\n" +
			"B : {false}?=>('a'|'b')+ {System.out.println(\"token 2\");} ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aa", false);
		// "a" is ambig; can match both A, B.  Pred says match A twice
		assertEquals("token 1\ntoken 1\n", found);
	}

	@Test public void testGatedPred2() throws Exception {
		String grammar =
			"grammar foo;\n" +
			"@lexer::members {boolean sig=false;}\n"+
			"a : (A|B)+ ;\n" +
			"A : 'a' {System.out.print(\"A\"); sig=true;} ;\n" +
			"B : 'b' ;\n" +
			"C : {sig}?=> ('a'|'b') {System.out.print(\"C\");} ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aa", false);
		assertEquals("AC\n", found);
	}

	@Test public void testPredWithActionTranslation() throws Exception {
		String grammar =
			"grammar foo;\n" +
			"a : b[2] ;\n" +
			"b[int i]\n" +
			"  : {$i==1}?   'a' {System.out.println(\"alt 1\");}\n" +
			"  | {$b.i==2}? 'a' {System.out.println(\"alt 2\");}\n" +
			"  ;\n";
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "aa", false);
		assertEquals("alt 2\n", found);
	}

	@Test public void testPredicatesOnEOTTarget() throws Exception {
		String grammar =
			"grammar foo; \n" +
			"@lexer::members {boolean p=true, q=false;}" +
			"a : B ;\n" +
			"A: '</'; \n" +
			"B: {p}? '<!' {System.out.println(\"B\");};\n" +
			"C: {q}? '<' {System.out.println(\"C\");}; \n" +
			"D: '<';\n" ;
		String found = execParser("foo.g", grammar, "fooParser", "fooLexer",
				    "a", "<!", false);
		assertEquals("B\n", found);
	}


	// S U P P O R T

	public void _test() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a :  ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {channel=99;} ;\n";
		String found = execParser("t.g", grammar, "T", "TLexer",
				    "a", "abc 34", false);
		assertEquals("\n", found);
	}

}
