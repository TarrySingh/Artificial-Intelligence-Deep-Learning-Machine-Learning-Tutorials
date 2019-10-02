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
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestASTConstruction extends BaseTest {

    /** Public default constructor used by TestRig */
    public TestASTConstruction() {
    }

	@Test public void testA() throws Exception {
		Grammar g = new Grammar(
				"parser grammar P;\n"+
				"a : A;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT A <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testNakeRulePlusInLexer() throws Exception {
		Grammar g = new Grammar(
				"lexer grammar P;\n"+
				"A : B+;\n" +
				"B : 'a';");
		String expecting =
			" ( rule A ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT B <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("A").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testRulePlus() throws Exception {
		Grammar g = new Grammar(
				"parser grammar P;\n"+
				"a : (b)+;\n" +
				"b : B;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT b <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testNakedRulePlus() throws Exception {
		Grammar g = new Grammar(
				"parser grammar P;\n"+
				"a : b+;\n" +
				"b : B;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT b <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testRuleOptional() throws Exception {
		Grammar g = new Grammar(
				"parser grammar P;\n"+
				"a : (b)?;\n" +
				"b : B;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( ? ( BLOCK ( ALT b <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testNakedRuleOptional() throws Exception {
		Grammar g = new Grammar(
				"parser grammar P;\n"+
				"a : b?;\n" +
				"b : B;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( ? ( BLOCK ( ALT b <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testRuleStar() throws Exception {
		Grammar g = new Grammar(
				"parser grammar P;\n"+
				"a : (b)*;\n" +
				"b : B;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT b <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testNakedRuleStar() throws Exception {
		Grammar g = new Grammar(
				"parser grammar P;\n"+
				"a : b*;\n" +
				"b : B;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT b <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testCharStar() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : 'a'*;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT 'a' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testCharStarInLexer() throws Exception {
		Grammar g = new Grammar(
				"lexer grammar P;\n"+
				"B : 'b'*;");
		String expecting =
			" ( rule B ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT 'b' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("B").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testStringStar() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : 'while'*;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT 'while' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testStringStarInLexer() throws Exception {
		Grammar g = new Grammar(
				"lexer grammar P;\n"+
				"B : 'while'*;");
		String expecting =
			" ( rule B ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT 'while' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("B").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testCharPlus() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : 'a'+;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT 'a' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testCharPlusInLexer() throws Exception {
		Grammar g = new Grammar(
				"lexer grammar P;\n"+
				"B : 'b'+;");
		String expecting =
			" ( rule B ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT 'b' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("B").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testCharOptional() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : 'a'?;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( ? ( BLOCK ( ALT 'a' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testCharOptionalInLexer() throws Exception {
		Grammar g = new Grammar(
				"lexer grammar P;\n"+
				"B : 'b'?;");
		String expecting =
			" ( rule B ARG RET scope ( BLOCK ( ALT ( ? ( BLOCK ( ALT 'b' <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("B").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testCharRangePlus() throws Exception {
		Grammar g = new Grammar(
				"lexer grammar P;\n"+
				"ID : 'a'..'z'+;");
		String expecting =
			" ( rule ID ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT ( .. 'a' 'z' ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("ID").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testLabel() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x=ID;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( = x ID ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testLabelOfOptional() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x=ID?;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( ? ( BLOCK ( ALT ( = x ID ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testLabelOfClosure() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x=ID*;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT ( = x ID ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testRuleLabel() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x=b;\n" +
				"b : ID;\n");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( = x b ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testSetLabel() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x=(A|B);\n");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( = x ( BLOCK ( ALT A <end-of-alt> ) ( ALT B <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testNotSetLabel() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x=~(A|B);\n");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( = x ( ~ ( BLOCK ( ALT A <end-of-alt> ) ( ALT B <end-of-alt> ) <end-of-block> ) ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testNotSetListLabel() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x+=~(A|B);\n");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( += x ( ~ ( BLOCK ( ALT A <end-of-alt> ) ( ALT B <end-of-alt> ) <end-of-block> ) ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testNotSetListLabelInLoop() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x+=~(A|B)+;\n");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT ( += x ( ~ ( BLOCK ( ALT A <end-of-alt> ) ( ALT B <end-of-alt> ) <end-of-block> ) ) ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testRuleLabelOfPositiveClosure() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x=b+;\n" +
				"b : ID;\n");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT ( = x b ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testListLabelOfClosure() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x+=ID*;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT ( += x ID ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testListLabelOfClosure2() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n"+
				"a : x+='int'*;");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( * ( BLOCK ( ALT ( += x 'int' ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testRuleListLabelOfPositiveClosure() throws Exception {
		Grammar g = new Grammar(
				"grammar P;\n" +
				"options {output=AST;}\n"+
				"a : x+=b+;\n" +
				"b : ID;\n");
		String expecting =
			" ( rule a ARG RET scope ( BLOCK ( ALT ( + ( BLOCK ( ALT ( += x b ) <end-of-alt> ) <end-of-block> ) ) <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("a").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testRootTokenInStarLoop() throws Exception {
		Grammar g = new Grammar(
				"grammar Expr;\n" +
				"options { backtrack=true; }\n" +
				"a : ('*'^)* ;\n");  // bug: the synpred had nothing in it
		String expecting =
			" ( rule synpred1_Expr ARG RET scope ( BLOCK ( ALT '*' <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("synpred1_Expr").tree.toStringTree();
		assertEquals(expecting, found);
	}

	@Test public void testActionInStarLoop() throws Exception {
		Grammar g = new Grammar(
				"grammar Expr;\n" +
				"options { backtrack=true; }\n" +
				"a : ({blort} 'x')* ;\n");  // bug: the synpred had nothing in it
		String expecting =
			" ( rule synpred1_Expr ARG RET scope ( BLOCK ( ALT blort 'x' <end-of-alt> ) <end-of-block> ) <end-of-rule> )";
		String found = g.getRule("synpred1_Expr").tree.toStringTree();
		assertEquals(expecting, found);
	}

}
