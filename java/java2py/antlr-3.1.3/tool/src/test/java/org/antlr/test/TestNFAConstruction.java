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

import org.antlr.analysis.State;
import org.antlr.tool.FASerializer;
import org.antlr.tool.Grammar;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestNFAConstruction extends BaseTest {

	/** Public default constructor used by TestRig */
	public TestNFAConstruction() {
	}

	@Test public void testA() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-A->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAB() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A B ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-A->.s3\n" +
			".s3-B->.s4\n" +
			".s4->:s5\n" +
			":s5-EOF->.s6\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAorB() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A | B {;} ;");
		/* expecting (0)--Ep-->(1)--Ep-->(2)--A-->(3)--Ep-->(4)--Ep-->(5,end)
										|                            ^
									   (6)--Ep-->(7)--B-->(8)--------|
				 */
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s1->.s7\n" +
			".s10->.s4\n" +
			".s2-A->.s3\n" +
			".s3->.s4\n" +
			".s4->:s5\n" +
			".s7->.s8\n" +
			".s8-B->.s9\n" +
			".s9-{}->.s10\n" +
			":s5-EOF->.s6\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testRangeOrRange() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : ('a'..'c' 'h' | 'q' 'j'..'l') ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10-'q'->.s11\n" +
			".s11-'j'..'l'->.s12\n" +
			".s12->.s6\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3-'a'..'c'->.s4\n" +
			".s4-'h'->.s5\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s10\n" +
			":s7-<EOT>->.s8\n";
		checkRule(g, "A", expecting);
	}

	@Test public void testRange() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : 'a'..'c' ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-'a'..'c'->.s3\n" +
			".s3->:s4\n" +
			":s4-<EOT>->.s5\n";
		checkRule(g, "A", expecting);
	}

	@Test public void testCharSetInParser() throws Exception {
		Grammar g = new Grammar(
			"grammar P;\n"+
			"a : A|'b' ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-A..'b'->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testABorCD() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A B | C D;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s1->.s8\n" +
			".s10-D->.s11\n" +
			".s11->.s5\n" +
			".s2-A->.s3\n" +
			".s3-B->.s4\n" +
			".s4->.s5\n" +
			".s5->:s6\n" +
			".s8->.s9\n" +
			".s9-C->.s10\n" +
			":s6-EOF->.s7\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testbA() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b A ;\n"+
			"b : B ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4->.s5\n" +
			".s5-B->.s6\n" +
			".s6->:s7\n" +
			".s8-A->.s9\n" +
			".s9->:s10\n" +
			":s10-EOF->.s11\n" +
			":s7->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testbA_bC() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b A ;\n"+
			"b : B ;\n"+
			"c : b C;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s12->.s13\n" +
			".s13-C->.s14\n" +
			".s14->:s15\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4->.s5\n" +
			".s5-B->.s6\n" +
			".s6->:s7\n" +
			".s8-A->.s9\n" +
			".s9->:s10\n" +
			":s10-EOF->.s11\n" +
			":s15-EOF->.s16\n" +
			":s7->.s12\n" +
			":s7->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAorEpsilon() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A | ;");
		/* expecting (0)--Ep-->(1)--Ep-->(2)--A-->(3)--Ep-->(4)--Ep-->(5,end)
										|                            ^
									   (6)--Ep-->(7)--Ep-->(8)-------|
				 */
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s1->.s7\n" +
			".s2-A->.s3\n" +
			".s3->.s4\n" +
			".s4->:s5\n" +
			".s7->.s8\n" +
			".s8->.s9\n" +
			".s9->.s4\n" +
			":s5-EOF->.s6\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAOptional() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A)?;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s8\n" +
			".s3-A->.s4\n" +
			".s4->.s5\n" +
			".s5->:s6\n" +
			".s8->.s5\n" +
			":s6-EOF->.s7\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testNakedAoptional() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A?;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s8\n" +
			".s3-A->.s4\n" +
			".s4->.s5\n" +
			".s5->:s6\n" +
			".s8->.s5\n" +
			":s6-EOF->.s7\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAorBthenC() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A | B) C;");
		/* expecting

				(0)--Ep-->(1)--Ep-->(2)--A-->(3)--Ep-->(4)--Ep-->(5)--C-->(6)--Ep-->(7,end)
						   |                            ^
						  (8)--Ep-->(9)--B-->(10)-------|
				 */
	}

	@Test public void testAplus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A)+;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testNakedAplus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A+;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAplusNonGreedy() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : (options {greedy=false;}:'0'..'9')+ ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4-'0'..'9'->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			":s7-<EOT>->.s8\n";
		checkRule(g, "A", expecting);
	}

	@Test public void testAorBplus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A | B{action})+ ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->.s11\n" +
			".s11-B->.s12\n" +
			".s12-{}->.s13\n" +
			".s13->.s6\n" +
			".s2->.s3\n" +
			".s3->.s10\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s6\n" +
			".s6->.s3\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAorBorEmptyPlus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A | B | )+ ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->.s11\n" +
			".s10->.s13\n" +
			".s11-B->.s12\n" +
			".s12->.s6\n" +
			".s13->.s14\n" +
			".s14->.s15\n" +
			".s15->.s6\n" +
			".s2->.s3\n" +
			".s3->.s10\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s6\n" +
			".s6->.s3\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAStar() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A)*;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s6\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testNestedAstar() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A*)*;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->:s11\n" +
			".s13->.s8\n" +
			".s14->.s10\n" +
			".s2->.s14\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4->.s13\n" +
			".s4->.s5\n" +
			".s5->.s6\n" +
			".s6-A->.s7\n" +
			".s7->.s5\n" +
			".s7->.s8\n" +
			".s8->.s9\n" +
			".s9->.s10\n" +
			".s9->.s3\n" +
			":s11-EOF->.s12\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testPlusNestedInStar() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A+)*;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->:s11\n" +
			".s13->.s10\n" +
			".s2->.s13\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4->.s5\n" +
			".s5->.s6\n" +
			".s6-A->.s7\n" +
			".s7->.s5\n" +
			".s7->.s8\n" +
			".s8->.s9\n" +
			".s9->.s10\n" +
			".s9->.s3\n" +
			":s11-EOF->.s12\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testStarNestedInPlus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A*)+;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->:s11\n" +
			".s13->.s8\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4->.s13\n" +
			".s4->.s5\n" +
			".s5->.s6\n" +
			".s6-A->.s7\n" +
			".s7->.s5\n" +
			".s7->.s8\n" +
			".s8->.s9\n" +
			".s9->.s10\n" +
			".s9->.s3\n" +
			":s11-EOF->.s12\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testNakedAstar() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A*;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s6\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAorBstar() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : (A | B{action})* ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->.s11\n" +
			".s11-B->.s12\n" +
			".s12-{}->.s13\n" +
			".s13->.s6\n" +
			".s14->.s7\n" +
			".s2->.s14\n" +
			".s2->.s3\n" +
			".s3->.s10\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s6\n" +
			".s6->.s3\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAorBOptionalSubrule() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : ( A | B )? ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s8\n" +
			".s3-A..B->.s4\n" +
			".s4->.s5\n" +
			".s5->:s6\n" +
			".s8->.s5\n" +
			":s6-EOF->.s7\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testPredicatedAorB() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? A | {p2}? B ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s1->.s8\n" +
			".s10-B->.s11\n" +
			".s11->.s5\n" +
			".s2-{p1}?->.s3\n" +
			".s3-A->.s4\n" +
			".s4->.s5\n" +
			".s5->:s6\n" +
			".s8->.s9\n" +
			".s9-{p2}?->.s10\n" +
			":s6-EOF->.s7\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testMultiplePredicates() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? {p1a}? A | {p2}? B | {p3} b;\n" +
			"b : {p4}? B ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s1->.s9\n" +
			".s10-{p2}?->.s11\n" +
			".s11-B->.s12\n" +
			".s12->.s6\n" +
			".s13->.s14\n" +
			".s14-{}->.s15\n" +
			".s15->.s16\n" +
			".s16->.s17\n" +
			".s17->.s18\n" +
			".s18-{p4}?->.s19\n" +
			".s19-B->.s20\n" +
			".s2-{p1}?->.s3\n" +
			".s20->:s21\n" +
			".s22->.s6\n" +
			".s3-{p1a}?->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s10\n" +
			".s9->.s13\n" +
			":s21->.s22\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testSets() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : ( A | B )+ ;\n" +
			"b : ( A | B{;} )+ ;\n" +
			"c : (A|B) (A|B) ;\n" +
			"d : ( A | B )* ;\n" +
			"e : ( A | B )? ;");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4-A..B->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
		expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->.s11\n" +
			".s11-B->.s12\n" +
			".s12-{}->.s13\n" +
			".s13->.s6\n" +
			".s2->.s3\n" +
			".s3->.s10\n" +
			".s3->.s4\n" +
			".s4-A->.s5\n" +
			".s5->.s6\n" +
			".s6->.s3\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "b", expecting);
		expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-A..B->.s3\n" +
			".s3-A..B->.s4\n" +
			".s4->:s5\n" +
			":s5-EOF->.s6\n";
		checkRule(g, "c", expecting);
		expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3->.s4\n" +
			".s4-A..B->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s6\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "d", expecting);
		expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s8\n" +
			".s3-A..B->.s4\n" +
			".s4->.s5\n" +
			".s5->:s6\n" +
			".s8->.s5\n" +
			":s6-EOF->.s7\n";
		checkRule(g, "e", expecting);
	}

	@Test public void testNotSet() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"tokens { A; B; C; }\n"+
			"a : ~A ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-B..C->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);

		String expectingGrammarStr =
			"1:8: parser grammar P;\n" +
			"a : ~ A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testNotSingletonBlockSet() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"tokens { A; B; C; }\n"+
			"a : ~(A) ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-B..C->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);

		String expectingGrammarStr =
			"1:8: parser grammar P;\n" +
			"a : ~ ( A ) ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testNotCharSet() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : ~'3' ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-{'\\u0000'..'2', '4'..'\\uFFFF'}->.s3\n" +
			".s3->:s4\n" +
			":s4-<EOT>->.s5\n";
		checkRule(g, "A", expecting);

		String expectingGrammarStr =
			"1:7: lexer grammar P;\n" +
			"A : ~ '3' ;\n"+
			"Tokens : A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testNotBlockSet() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : ~('3'|'b') ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-{'\\u0000'..'2', '4'..'a', 'c'..'\\uFFFF'}->.s3\n" +
			".s3->:s4\n" +
			":s4-<EOT>->.s5\n";
		checkRule(g, "A", expecting);

		String expectingGrammarStr =
			"1:7: lexer grammar P;\n" +
			"A : ~ ( '3' | 'b' ) ;\n" +
			"Tokens : A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testNotSetLoop() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : ~('3')* ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3->.s4\n" +
			".s4-{'\\u0000'..'2', '4'..'\\uFFFF'}->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s6\n" +
			":s7-<EOT>->.s8\n";
		checkRule(g, "A", expecting);

		String expectingGrammarStr =
			"1:7: lexer grammar P;\n" +
			"A : (~ ( '3' ) )* ;\n" +
			"Tokens : A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testNotBlockSetLoop() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : ~('3'|'b')* ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3->.s4\n" +
			".s4-{'\\u0000'..'2', '4'..'a', 'c'..'\\uFFFF'}->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s6\n" +
			":s7-<EOT>->.s8\n";
		checkRule(g, "A", expecting);

		String expectingGrammarStr =
			"1:7: lexer grammar P;\n" +
			"A : (~ ( '3' | 'b' ) )* ;\n" +
			"Tokens : A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testSetsInCombinedGrammarSentToLexer() throws Exception {
		// not sure this belongs in this test suite, but whatever.
		Grammar g = new Grammar(
			"grammar t;\n"+
			"A : '{' ~('}')* '}';\n");
		String result = g.getLexerGrammar();
		String expecting =
			"lexer grammar t;\n" +
			"\n" +
			"// $ANTLR src \"<string>\" 2\n"+
			"A : '{' ~('}')* '}';\n";
		assertEquals(result, expecting);
	}

	@Test public void testLabeledNotSet() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"tokens { A; B; C; }\n"+
			"a : t=~A ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-B..C->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);

		String expectingGrammarStr =
			"1:8: parser grammar P;\n" +
			"a : t=~ A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testLabeledNotCharSet() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : t=~'3' ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-{'\\u0000'..'2', '4'..'\\uFFFF'}->.s3\n" +
			".s3->:s4\n" +
			":s4-<EOT>->.s5\n";
		checkRule(g, "A", expecting);

		String expectingGrammarStr =
			"1:7: lexer grammar P;\n" +
			"A : t=~ '3' ;\n"+
			"Tokens : A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testLabeledNotBlockSet() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : t=~('3'|'b') ;\n");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-{'\\u0000'..'2', '4'..'a', 'c'..'\\uFFFF'}->.s3\n" +
			".s3->:s4\n" +
			":s4-<EOT>->.s5\n";
		checkRule(g, "A", expecting);

		String expectingGrammarStr =
			"1:7: lexer grammar P;\n" +
			"A : t=~ ( '3' | 'b' ) ;\n" +
			"Tokens : A ;";
		assertEquals(expectingGrammarStr, g.toString());
	}

	@Test public void testEscapedCharLiteral() throws Exception {
		Grammar g = new Grammar(
			"grammar P;\n"+
			"a : '\\n';");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-'\\n'->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testEscapedStringLiteral() throws Exception {
		Grammar g = new Grammar(
			"grammar P;\n"+
			"a : 'a\\nb\\u0030c\\'';");
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-'a\\nb\\u0030c\\''->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);
	}

	// AUTO BACKTRACKING STUFF

	@Test public void testAutoBacktracking_RuleBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : 'a'{;}|'b';"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s1->.s9\n" +
			".s10-'b'->.s11\n" +
			".s11->.s6\n" +
			".s2-{synpred1_t}?->.s3\n" +
			".s3-'a'->.s4\n" +
			".s4-{}->.s5\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s10\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_RuleSetBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : 'a'|'b';"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-'a'..'b'->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_SimpleBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'{;}|'b') ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->.s11\n" +
			".s11-'b'->.s12\n" +
			".s12->.s7\n" +
			".s2->.s10\n" +
			".s2->.s3\n" +
			".s3-{synpred1_t}?->.s4\n" +
			".s4-'a'->.s5\n" +
			".s5-{}->.s6\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_SetBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'|'b') ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2-'a'..'b'->.s3\n" +
			".s3->:s4\n" +
			":s4-EOF->.s5\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_StarBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'{;}|'b')* ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s12->.s13\n" +
			".s13-{synpred2_t}?->.s14\n" +
			".s14-'b'->.s15\n" +
			".s15->.s8\n" +
			".s16->.s9\n" +
			".s2->.s16\n" +
			".s2->.s3\n" +
			".s3->.s12\n" +
			".s3->.s4\n" +
			".s4-{synpred1_t}?->.s5\n" +
			".s5-'a'->.s6\n" +
			".s6-{}->.s7\n" +
			".s7->.s8\n" +
			".s8->.s3\n" +
			".s8->.s9\n" +
			".s9->:s10\n" +
			":s10-EOF->.s11\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_StarSetBlock_IgnoresPreds() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'|'b')* ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3->.s4\n" +
			".s4-'a'..'b'->.s5\n" +
			".s5->.s3\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s6\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_StarSetBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'|'b'{;})* ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s11->.s12\n" +
			".s12-{synpred2_t}?->.s13\n" +
			".s13-'b'->.s14\n" +
			".s14-{}->.s15\n" +
			".s15->.s7\n" +
			".s16->.s8\n" +
			".s2->.s16\n" +
			".s2->.s3\n" +
			".s3->.s11\n" +
			".s3->.s4\n" +
			".s4-{synpred1_t}?->.s5\n" +
			".s5-'a'->.s6\n" +
			".s6->.s7\n" +
			".s7->.s3\n" +
			".s7->.s8\n" +
			".s8->:s9\n" +
			":s9-EOF->.s10\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_StarBlock1Alt() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a')* ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->.s7\n" +
			".s2->.s10\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4-{synpred1_t}?->.s5\n" +
			".s5-'a'->.s6\n" +
			".s6->.s3\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_PlusBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'{;}|'b')+ ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s12->.s13\n" +
			".s13-{synpred2_t}?->.s14\n" +
			".s14-'b'->.s15\n" +
			".s15->.s8\n" +
			".s2->.s3\n" +
			".s3->.s12\n" +
			".s3->.s4\n" +
			".s4-{synpred1_t}?->.s5\n" +
			".s5-'a'->.s6\n" +
			".s6-{}->.s7\n" +
			".s7->.s8\n" +
			".s8->.s3\n" +
			".s8->.s9\n" +
			".s9->:s10\n" +
			":s10-EOF->.s11\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_PlusSetBlock() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'|'b'{;})+ ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s11->.s12\n" +
			".s12-{synpred2_t}?->.s13\n" +
			".s13-'b'->.s14\n" +
			".s14-{}->.s15\n" +
			".s15->.s7\n" +
			".s2->.s3\n" +
			".s3->.s11\n" +
			".s3->.s4\n" +
			".s4-{synpred1_t}?->.s5\n" +
			".s5-'a'->.s6\n" +
			".s6->.s7\n" +
			".s7->.s3\n" +
			".s7->.s8\n" +
			".s8->:s9\n" +
			":s9-EOF->.s10\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_PlusBlock1Alt() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a')+ ;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s3->.s4\n" +
			".s4-{synpred1_t}?->.s5\n" +
			".s5-'a'->.s6\n" +
			".s6->.s3\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_OptionalBlock2Alts() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a'{;}|'b')?;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s10->.s11\n" +
			".s10->.s14\n" +
			".s11-{synpred2_t}?->.s12\n" +
			".s12-'b'->.s13\n" +
			".s13->.s7\n" +
			".s14->.s7\n" +
			".s2->.s10\n" +
			".s2->.s3\n" +
			".s3-{synpred1_t}?->.s4\n" +
			".s4-'a'->.s5\n" +
			".s5-{}->.s6\n" +
			".s6->.s7\n" +
			".s7->:s8\n" +
			":s8-EOF->.s9\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_OptionalBlock1Alt() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a')?;"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s2->.s3\n" +
			".s2->.s9\n" +
			".s3-{synpred1_t}?->.s4\n" +
			".s4-'a'->.s5\n" +
			".s5->.s6\n" +
			".s6->:s7\n" +
			".s9->.s6\n" +
			":s7-EOF->.s8\n";
		checkRule(g, "a", expecting);
	}

	@Test public void testAutoBacktracking_ExistingPred() throws Exception {
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {backtrack=true;}\n"+
			"a : ('a')=> 'a' | 'b';"
		);
		String expecting =
			".s0->.s1\n" +
			".s1->.s2\n" +
			".s1->.s8\n" +
			".s10->.s5\n" +
			".s2-{synpred1_t}?->.s3\n" +
			".s3-'a'->.s4\n" +
			".s4->.s5\n" +
			".s5->:s6\n" +
			".s8->.s9\n" +
			".s9-'b'->.s10\n" +
			":s6-EOF->.s7\n";
		checkRule(g, "a", expecting);
	}

	private void checkRule(Grammar g, String rule, String expecting)
	{
		g.buildNFA();
		State startState = g.getRuleStartState(rule);
		FASerializer serializer = new FASerializer(g);
		String result = serializer.serialize(startState);

		//System.out.print(result);
		assertEquals(expecting, result);
	}

}
