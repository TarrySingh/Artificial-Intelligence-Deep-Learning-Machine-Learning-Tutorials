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

import org.antlr.analysis.DFA;
import org.antlr.analysis.DecisionProbe;
import org.antlr.misc.BitSet;
import org.antlr.tool.*;
import org.antlr.Tool;
import org.antlr.codegen.CodeGenerator;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.*;

public class TestDFAConversion extends BaseTest {

	@Test public void testA() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A C | B;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testAB_or_AC() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A B | A C;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->:s2=>1\n" +
			".s1-C->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testAB_or_AC_k2() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"options {k=2;}\n"+
			"a : A B | A C;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->:s2=>1\n" +
			".s1-C->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testAB_or_AC_k1() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"options {k=1;}\n"+
			"a : A B | A C;");
		String expecting =
			".s0-A->:s1=>1\n";
		int[] unreachableAlts = new int[] {2};
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "A" ;
		int[] danglingAlts = new int[] {2};
		int numWarnings = 2; // ambig upon A
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testselfRecurseNonDet() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : A a X | A a Y;");
		List altsWithRecursion = Arrays.asList(new Object[] {1,2});
		assertNonLLStar(g, altsWithRecursion);
	}

	@Test public void testRecursionOverflow() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a Y | A A A A A X ;\n" + // force recursion past m=4
			"a : A a | Q;");
		List expectedTargetRules = Arrays.asList(new Object[] {"a"});
		int expectedAlt = 1;
		assertRecursionOverflow(g, expectedTargetRules, expectedAlt);
	}

	@Test public void testRecursionOverflow2() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a Y | A+ X ;\n" + // force recursion past m=4
			"a : A a | Q;");
		List expectedTargetRules = Arrays.asList(new Object[] {"a"});
		int expectedAlt = 1;
		assertRecursionOverflow(g, expectedTargetRules, expectedAlt);
	}

	@Test public void testRecursionOverflowWithPredOk() throws Exception {
		// overflows with k=*, but resolves with pred
		// no warnings/errors
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : (a Y)=> a Y | A A A A A X ;\n" + // force recursion past m=4
			"a : A a | Q;");
		String expecting =
			".s0-A->.s1\n" +
			".s0-Q&&{synpred1_t}?->:s11=>1\n" +
			".s1-A->.s2\n" +
			".s1-Q&&{synpred1_t}?->:s10=>1\n" +
			".s2-A->.s3\n" +
			".s2-Q&&{synpred1_t}?->:s9=>1\n" +
			".s3-A->.s4\n" +
			".s3-Q&&{synpred1_t}?->:s8=>1\n" +
			".s4-A->.s5\n" +
			".s4-Q&&{synpred1_t}?->:s6=>1\n" +
			".s5-{synpred1_t}?->:s6=>1\n" +
			".s5-{true}?->:s7=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testRecursionOverflowWithPredOk2() throws Exception {
		// must predict Z w/o predicate
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : (a Y)=> a Y | A A A A A X | Z;\n" + // force recursion past m=4
			"a : A a | Q;");
		String expecting =
			".s0-A->.s1\n" +
			".s0-Q&&{synpred1_t}?->:s11=>1\n" +
			".s0-Z->:s12=>3\n" +
			".s1-A->.s2\n" +
			".s1-Q&&{synpred1_t}?->:s10=>1\n" +
			".s2-A->.s3\n" +
			".s2-Q&&{synpred1_t}?->:s9=>1\n" +
			".s3-A->.s4\n" +
			".s3-Q&&{synpred1_t}?->:s8=>1\n" +
			".s4-A->.s5\n" +
			".s4-Q&&{synpred1_t}?->:s6=>1\n" +
			".s5-{synpred1_t}?->:s6=>1\n" +
			".s5-{true}?->:s7=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testCannotSeePastRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"x   : y X\n" +
			"    | y Y\n" +
			"    ;\n" +
			"y   : L y R\n" +
			"    | B\n" +
			"    ;");
		List altsWithRecursion = Arrays.asList(new Object[] {1,2});
		assertNonLLStar(g, altsWithRecursion);
	}

	@Test public void testSynPredResolvesRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"x   : (y X)=> y X\n" +
			"    | y Y\n" +
			"    ;\n" +
			"y   : L y R\n" +
			"    | B\n" +
			"    ;");
		String expecting =
			".s0-B->.s4\n" +
			".s0-L->.s1\n" +
			".s1-{synpred1_t}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n" +
			".s4-{synpred1_t}?->:s2=>1\n" +
			".s4-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testSemPredResolvesRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"x   : {p}? y X\n" +
			"    | y Y\n" +
			"    ;\n" +
			"y   : L y R\n" +
			"    | B\n" +
			"    ;");
		String expecting =
			".s0-B->.s4\n" +
			".s0-L->.s1\n" +
			".s1-{p}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n" +
			".s4-{p}?->:s2=>1\n" +
			".s4-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testSemPredResolvesRecursion2() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"x\n" +
			"options {k=1;}\n" +
			"   : {p}? y X\n" +
			"    | y Y\n" +
			"    ;\n" +
			"y   : L y R\n" +
			"    | B\n" +
			"    ;");
		String expecting =
			".s0-B->.s4\n" +
			".s0-L->.s1\n" +
			".s1-{p}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n" +
			".s4-{p}?->:s2=>1\n" +
			".s4-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testSemPredResolvesRecursion3() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"x\n" +
			"options {k=2;}\n" + // just makes bigger DFA
			"   : {p}? y X\n" +
			"    | y Y\n" +
			"    ;\n" +
			"y   : L y R\n" +
			"    | B\n" +
			"    ;");
		String expecting =
			".s0-B->.s6\n" +
			".s0-L->.s1\n" +
			".s1-B->.s5\n" +
			".s1-L->.s2\n" +
			".s2-{p}?->:s3=>1\n" +
			".s2-{true}?->:s4=>2\n" +
			".s5-{p}?->:s3=>1\n" +
			".s5-{true}?->:s4=>2\n" +
			".s6-X->:s3=>1\n" +
			".s6-Y->:s4=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testSynPredResolvesRecursion2() throws Exception {
		// k=* fails and it retries/succeeds with k=1 silently
		// because of predicate
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"statement\n" +
			"    :     (reference ASSIGN)=> reference ASSIGN expr\n" +
			"    |     expr\n" +
			"    ;\n" +
			"expr:     reference\n" +
			"    |     INT\n" +
			"    |     FLOAT\n" +
			"    ;\n" +
			"reference\n" +
			"    :     ID L argument_list R\n" +
			"    ;\n" +
			"argument_list\n" +
			"    :     expr COMMA expr\n" +
			"    ;");
		String expecting =
			".s0-ID->.s1\n" +
			".s0-INT..FLOAT->:s3=>2\n" +
			".s1-{synpred1_t}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testSynPredResolvesRecursion3() throws Exception {
		// No errors with k=1; don't try k=* first
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"statement\n" +
			"options {k=1;}\n" +
			"    :     (reference ASSIGN)=> reference ASSIGN expr\n" +
			"    |     expr\n" +
			"    ;\n" +
			"expr:     reference\n" +
			"    |     INT\n" +
			"    |     FLOAT\n" +
			"    ;\n" +
			"reference\n" +
			"    :     ID L argument_list R\n" +
			"    ;\n" +
			"argument_list\n" +
			"    :     expr COMMA expr\n" +
			"    ;");
		String expecting =
			".s0-ID->.s1\n" +
			".s0-INT..FLOAT->:s3=>2\n" +
			".s1-{synpred1_t}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testSynPredResolvesRecursion4() throws Exception {
		// No errors with k=2; don't try k=* first
		// Should be ok like k=1 'except bigger DFA
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"statement\n" +
			"options {k=2;}\n" +
			"    :     (reference ASSIGN)=> reference ASSIGN expr\n" +
			"    |     expr\n" +
			"    ;\n" +
			"expr:     reference\n" +
			"    |     INT\n" +
			"    |     FLOAT\n" +
			"    ;\n" +
			"reference\n" +
			"    :     ID L argument_list R\n" +
			"    ;\n" +
			"argument_list\n" +
			"    :     expr COMMA expr\n" +
			"    ;");
		String expecting =
			".s0-ID->.s1\n" +
			".s0-INT..FLOAT->:s4=>2\n" +
			".s1-L->.s2\n" +
			".s2-{synpred1_t}?->:s3=>1\n" +
			".s2-{true}?->:s4=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testSynPredResolvesRecursionInLexer() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A :     (B ';')=> B ';'\n" +
			"  |     B '.'\n" +
			"  ;\n" +
			"fragment\n" +
			"B :     '(' B ')'\n" +
			"  |     'x'\n" +
			"  ;\n");
		String expecting =
			".s0-'('->.s1\n" +
			".s0-'x'->.s4\n" +
			".s1-{synpred1_t}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n" +
			".s4-{synpred1_t}?->:s2=>1\n" +
			".s4-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testAutoBacktrackResolvesRecursionInLexer() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"options {backtrack=true;}\n"+
			"A :     B ';'\n" +
			"  |     B '.'\n" +
			"  ;\n" +
			"fragment\n" +
			"B :     '(' B ')'\n" +
			"  |     'x'\n" +
			"  ;\n");
		String expecting =
			".s0-'('->.s1\n" +
			".s0-'x'->.s4\n" +
			".s1-{synpred1_t}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n" +
			".s4-{synpred1_t}?->:s2=>1\n" +
			".s4-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testAutoBacktrackResolvesRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"options {backtrack=true;}\n"+
			"x   : y X\n" +
			"    | y Y\n" +
			"    ;\n" +
			"y   : L y R\n" +
			"    | B\n" +
			"    ;");
		String expecting =
			".s0-B->.s4\n" +
			".s0-L->.s1\n" +
			".s1-{synpred1_t}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n" +
			".s4-{synpred1_t}?->:s2=>1\n" +
			".s4-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testselfRecurseNonDet2() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : P a P | P;");
		// nondeterministic from left edge
		String expecting =
			".s0-P->.s1\n" +
			".s1-EOF->:s2=>2\n"+
			".s1-P->:s3=>1\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "P P";
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testIndirectRecursionLoop() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : b X ;\n"+
			"b : a B ;\n");

		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		Set<Rule> leftRecursive = g.getLeftRecursiveRules();
		Set expectedRules =
			new HashSet() {{add("a"); add("b");}};
		assertEquals(expectedRules, ruleNames(leftRecursive));

		g.createLookaheadDFAs(false);

		Message msg = (Message)equeue.warnings.get(0);
		assertTrue("expecting left recursion cycles; found "+msg.getClass().getName(),
				    msg instanceof LeftRecursionCyclesMessage);
		LeftRecursionCyclesMessage cyclesMsg = (LeftRecursionCyclesMessage)msg;

		// cycle of [a, b]
		Collection result = cyclesMsg.cycles;
		Set expecting = new HashSet() {{add("a"); add("b");}};
		assertEquals(expecting, ruleNames2(result));
	}

	@Test public void testIndirectRecursionLoop2() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : i b X ;\n"+ // should see through i
			"b : a B ;\n" +
			"i : ;\n");

		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		Set leftRecursive = g.getLeftRecursiveRules();
		Set expectedRules =
			new HashSet() {{add("a"); add("b");}};
		assertEquals(expectedRules, ruleNames(leftRecursive));

		g.createLookaheadDFAs(false);

		Message msg = (Message)equeue.warnings.get(0);
		assertTrue("expecting left recursion cycles; found "+msg.getClass().getName(),
				    msg instanceof LeftRecursionCyclesMessage);
		LeftRecursionCyclesMessage cyclesMsg = (LeftRecursionCyclesMessage)msg;

		// cycle of [a, b]
		Collection result = cyclesMsg.cycles;
		Set expecting = new HashSet() {{add("a"); add("b");}};
		assertEquals(expecting, ruleNames2(result));
	}

	@Test public void testIndirectRecursionLoop3() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : i b X ;\n"+ // should see through i
			"b : a B ;\n" +
			"i : ;\n" +
			"d : e ;\n" +
			"e : d ;\n");

		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		Set leftRecursive = g.getLeftRecursiveRules();
		Set expectedRules =
			new HashSet() {{add("a"); add("b"); add("e"); add("d");}};
		assertEquals(expectedRules, ruleNames(leftRecursive));

		Message msg = (Message)equeue.warnings.get(0);
		assertTrue("expecting left recursion cycles; found "+msg.getClass().getName(),
				    msg instanceof LeftRecursionCyclesMessage);
		LeftRecursionCyclesMessage cyclesMsg = (LeftRecursionCyclesMessage)msg;

		// cycle of [a, b]
		Collection result = cyclesMsg.cycles;
		Set expecting = new HashSet() {{add("a"); add("b"); add("d"); add("e");}};
		assertEquals(expecting, ruleNames2(result));
	}

	@Test public void testifThenElse() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : IF s (E s)? | B;\n" +
			"slist: s SEMI ;");
		String expecting =
			".s0-E->:s1=>1\n" +
			".s0-SEMI->:s2=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "E";
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
		expecting =
			".s0-B->:s2=>2\n" +
			".s0-IF->:s1=>1\n";
		checkDecision(g, 2, expecting, null, null, null, null, 0);
	}

	@Test public void testifThenElseChecksStackSuffixConflict() throws Exception {
		// if you don't check stack soon enough, this finds E B not just E
		// as ambig input
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"slist: s SEMI ;\n"+
			"s : IF s el | B;\n" +
			"el: (E s)? ;\n");
		String expecting =
			".s0-E->:s1=>1\n" +
			".s0-SEMI->:s2=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "E";
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 2, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
		expecting =
			".s0-B->:s2=>2\n" +
			".s0-IF->:s1=>1\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

    @Test
    public void testInvokeRule() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : b A\n" +
			"  | b B\n" +
			"  | C\n" +
			"  ;\n" +
			"b : X\n" +
			"  ;\n");
		String expecting =
			".s0-C->:s4=>3\n" +
            ".s0-X->.s1\n" +
            ".s1-A->:s2=>1\n" +
            ".s1-B->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test
    public void testDoubleInvokeRuleLeftEdge() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : b X\n" +
			"  | b Y\n" +
			"  ;\n" +
			"b : c B\n" +
			"  | c\n" +
			"  ;\n" +
			"c : C ;\n");
		String expecting =
			".s0-C->.s1\n" +
            ".s1-B->.s2\n" +
            ".s1-X->:s4=>1\n" +
            ".s1-Y->:s3=>2\n" +
            ".s2-X->:s4=>1\n" +
            ".s2-Y->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
		expecting =
			".s0-C->.s1\n" +
            ".s1-B->:s2=>1\n" +
            ".s1-X..Y->:s3=>2\n";
		checkDecision(g, 2, expecting, null, null, null, null, 0);
	}

	@Test public void testimmediateTailRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : A a | A B;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-A->:s3=>1\n" +
			".s1-B->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test
    public void testAStar_immediateTailRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : A a | ;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-EOF->:s2=>2\n";
		int[] unreachableAlts = null; // without
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testNoStartRule() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A a | X;"); // single rule 'a' refers to itself; no start rule

		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		Message msg = (Message)equeue.warnings.get(0);
		assertTrue("expecting no start rules; found "+msg.getClass().getName(),
				   msg instanceof GrammarSemanticsMessage);
	}

	@Test
    public void testAStar_immediateTailRecursion2() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : A a | A ;");
		String expecting =
			".s0-A->.s1\n" +
            ".s1-A->:s2=>1\n" +
            ".s1-EOF->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testimmediateLeftRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : a A | B;");
		Set leftRecursive = g.getLeftRecursiveRules();
		Set expectedRules = new HashSet() {{add("a");}};
		assertEquals(expectedRules, ruleNames(leftRecursive));
	}

	@Test public void testIndirectLeftRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : b | A ;\n" +
			"b : c ;\n" +
			"c : a | C ;\n");
		Set leftRecursive = g.getLeftRecursiveRules();
		Set expectedRules = new HashSet() {{add("a"); add("b"); add("c");}};
		assertEquals(expectedRules, ruleNames(leftRecursive));
	}

	@Test public void testLeftRecursionInMultipleCycles() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
				"s : a x ;\n" +
				"a : b | A ;\n" +
				"b : c ;\n" +
				"c : a | C ;\n" +
				"x : y | X ;\n" +
				"y : x ;\n");
		Set leftRecursive = g.getLeftRecursiveRules();
		Set expectedRules =
			new HashSet() {{add("a"); add("b"); add("c"); add("x"); add("y");}};
		assertEquals(expectedRules, ruleNames(leftRecursive));
	}

	@Test public void testCycleInsideRuleDoesNotForceInfiniteRecursion() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a ;\n" +
			"a : (A|)+ B;\n");
		// before I added a visitedStates thing, it was possible to loop
		// forever inside of a rule if there was an epsilon loop.
		Set leftRecursive = g.getLeftRecursiveRules();
		Set expectedRules = new HashSet();
		assertEquals(expectedRules, leftRecursive);
	}

	// L O O P S

	@Test public void testAStar() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ( A )* ;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-EOF->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testAorBorCStar() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ( A | B | C )* ;");
		String expecting =
			".s0-A..C->:s1=>1\n" +
			".s0-EOF->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testAPlus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ( A )+ ;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-EOF->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0); // loopback decision
	}

	@Test public void testAPlusNonGreedyWhenDeterministic() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (options {greedy=false;}:A)+ ;\n");
		// should look the same as A+ since no ambiguity
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-EOF->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testAPlusNonGreedyWhenNonDeterministic() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (options {greedy=false;}:A)+ A+ ;\n");
		// should look the same as A+ since no ambiguity
		String expecting =
			".s0-A->:s1=>2\n"; // always chooses to exit
		int[] unreachableAlts = new int[] {1};
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "A";
		int[] danglingAlts = null;
		int numWarnings = 2;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testAPlusGreedyWhenNonDeterministic() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (options {greedy=true;}:A)+ A+ ;\n");
		// should look the same as A+ since no ambiguity
		String expecting =
			".s0-A->:s1=>1\n"; // always chooses to enter loop upon A
		int[] unreachableAlts = new int[] {2};
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "A";
		int[] danglingAlts = null;
		int numWarnings = 2;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testAorBorCPlus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ( A | B | C )+ ;");
		String expecting =
			".s0-A..C->:s1=>1\n" +
			".s0-EOF->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testAOptional() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ( A )? B ;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0); // loopback decision
	}

	@Test public void testAorBorCOptional() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ( A | B | C )? Z ;");
		String expecting =
			".s0-A..C->:s1=>1\n" +
			".s0-Z->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0); // loopback decision
	}

	// A R B I T R A R Y  L O O K A H E A D

	@Test
    public void testAStarBOrAStarC() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A)* B | (A)* C;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0); // loopback
		expecting =
			".s0-A->:s1=>1\n" +
			".s0-C->:s2=>2\n";
		checkDecision(g, 2, expecting, null, null, null, null, 0); // loopback
		expecting =
			".s0-A->.s1\n" +
            ".s0-B->:s2=>1\n" +
            ".s0-C->:s3=>2\n" +
            ".s1-A->.s1\n" +
            ".s1-B->:s2=>1\n" +
            ".s1-C->:s3=>2\n";
		checkDecision(g, 3, expecting, null, null, null, null, 0); // rule block
	}

	@Test
    public void testAStarBOrAPlusC() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A)* B | (A)+ C;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0); // loopback
		expecting =
			".s0-A->:s1=>1\n" +
			".s0-C->:s2=>2\n";
		checkDecision(g, 2, expecting, null, null, null, null, 0); // loopback
		expecting =
			".s0-A->.s1\n" +
            ".s0-B->:s2=>1\n" +
            ".s1-A->.s1\n" +
            ".s1-B->:s2=>1\n" +
            ".s1-C->:s3=>2\n";
		checkDecision(g, 3, expecting, null, null, null, null, 0); // rule block
	}


    @Test
    public void testAOrBPlusOrAPlus() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A|B)* X | (A)+ Y;");
		String expecting =
			".s0-A..B->:s1=>1\n" +
			".s0-X->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0); // loopback (A|B)*
		expecting =
			".s0-A->:s1=>1\n" +
			".s0-Y->:s2=>2\n";
		checkDecision(g, 2, expecting, null, null, null, null, 0); // loopback (A)+
		expecting =
			".s0-A->.s1\n" +
            ".s0-B..X->:s2=>1\n" +
            ".s1-A->.s1\n" +
            ".s1-B..X->:s2=>1\n" +
            ".s1-Y->:s3=>2\n";
		checkDecision(g, 3, expecting, null, null, null, null, 0); // rule
	}

	@Test public void testLoopbackAndExit() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A|B)+ B;");
		String expecting =
			".s0-A->:s2=>1\n" +
			".s0-B->.s1\n" +
			".s1-A..B->:s2=>1\n" +
			".s1-EOF->:s3=>2\n"; // sees A|B as a set
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testOptionalAltAndBypass() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A|B)? B;");
		String expecting =
			".s0-A->:s2=>1\n" +
			".s0-B->.s1\n" +
			".s1-B->:s2=>1\n" +
			".s1-EOF->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	// R E S O L V E  S Y N  C O N F L I C T S

	@Test public void testResolveLL1ByChoosingFirst() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A C | A C;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-C->:s2=>1\n";
		int[] unreachableAlts = new int[] {2};
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "A C";
		int[] danglingAlts = null;
		int numWarnings = 2;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testResolveLL2ByChoosingFirst() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A B | A B;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->:s2=>1\n";
		int[] unreachableAlts = new int[] {2};
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "A B";
		int[] danglingAlts = null;
		int numWarnings = 2;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testResolveLL2MixAlt() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A B | A C | A B | Z;");
		String expecting =
			".s0-A->.s1\n" +
			".s0-Z->:s4=>4\n" +
			".s1-B->:s2=>1\n" +
			".s1-C->:s3=>2\n";
		int[] unreachableAlts = new int[] {3};
		int[] nonDetAlts = new int[] {1,3};
		String ambigInput = "A B";
		int[] danglingAlts = null;
		int numWarnings = 2;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testIndirectIFThenElseStyleAmbig() throws Exception {
		// the (c)+ loopback is ambig because it could match "CASE"
		// by entering the loop or by falling out and ignoring (s)*
		// back falling back into (cg)* loop which stats over and
		// calls cg again.  Either choice allows it to get back to
		// the same node.  The software catches it as:
		// "avoid infinite closure computation emanating from alt 1
		// of ():27|2|[8 $]" where state 27 is the first alt of (c)+
		// and 8 is the first alt of the (cg)* loop.
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"s : stat ;\n" +
			"stat : LCURLY ( cg )* RCURLY | E SEMI  ;\n" +
			"cg : (c)+ (stat)* ;\n" +
			"c : CASE E ;\n");
		String expecting =
			".s0-CASE->:s2=>1\n" +
			".s0-LCURLY..E->:s1=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "CASE";
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 3, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	// S E T S

	@Test public void testComplement() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ~(A | B | C) | C {;} ;\n" +
			"b : X Y Z ;");
		String expecting =
			".s0-C->:s2=>2\n" +
			".s0-X..Z->:s1=>1\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testComplementToken() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ~C | C {;} ;\n" +
			"b : X Y Z ;");
		String expecting =
			".s0-C->:s2=>2\n" +
			".s0-X..Z->:s1=>1\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testComplementChar() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : ~'x' | 'x' {;} ;\n");
		String expecting =
			".s0-'x'->:s2=>2\n" +
			".s0-{'\\u0000'..'w', 'y'..'\\uFFFF'}->:s1=>1\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testComplementCharSet() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : ~(' '|'\t'|'x'|'y') | 'x';\n" + // collapse into single set
			"B : 'y' ;");
		String expecting =
			".s0-'y'->:s2=>2\n" +
			".s0-{'\\u0000'..'\\b', '\\n'..'\\u001F', '!'..'x', 'z'..'\\uFFFF'}->:s1=>1\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testNoSetCollapseWithActions() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A | B {foo}) | C;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testRuleAltsSetCollapse() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A | B | C ;"
		);
		String expecting = // still looks like block
			" ( grammar t ( rule a ARG RET scope ( BLOCK ( ALT A <end-of-alt> ) ( ALT B <end-of-alt> ) ( ALT C <end-of-alt> ) <end-of-block> ) <end-of-rule> ) )";
		assertEquals(expecting, g.getGrammarTree().toStringTree());
	}

	@Test public void testTokensRuleAltsDoNotCollapse() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : 'a';" +
			"B : 'b';\n"
		);
		String expecting =
			".s0-'a'->:s1=>1\n" +
			".s0-'b'->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	@Test public void testMultipleSequenceCollision() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"a : (A{;}|B)\n" +
			"  | (A{;}|B)\n" +
			"  | A\n" +
			"  ;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>1\n"; // not optimized because states are nondet
		int[] unreachableAlts = new int[] {2,3};
		int[] nonDetAlts = new int[] {1,2,3};
		String ambigInput = "A";
		int[] danglingAlts = null;
		int numWarnings = 3;
		checkDecision(g, 3, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
		/* There are 2 nondet errors, but the checkDecision only checks first one :(
		The "B" conflicting input is not checked except by virtue of the
		result DFA.
<string>:2:5: Decision can match input such as "A" using multiple alternatives:
alt 1 via NFA path 7,2,3
alt 2 via NFA path 14,9,10
alt 3 via NFA path 16,17
As a result, alternative(s) 2,3 were disabled for that input,
<string>:2:5: Decision can match input such as "B" using multiple alternatives:
alt 1 via NFA path 7,8,4,5
alt 2 via NFA path 14,15,11,12
As a result, alternative(s) 2 were disabled for that input
<string>:2:5: The following alternatives are unreachable: 2,3
*/
	}

	@Test public void testMultipleAltsSameSequenceCollision() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"a : type ID \n" +
			"  | type ID\n" +
			"  | type ID\n" +
			"  | type ID\n" +
			"  ;\n" +
			"\n" +
			"type : I | F;");
		// nondeterministic from left edge; no stop state
		String expecting =
			".s0-I..F->.s1\n" +
			".s1-ID->:s2=>1\n";
		int[] unreachableAlts = new int[] {2,3,4};
		int[] nonDetAlts = new int[] {1,2,3,4};
		String ambigInput = "I..F ID";
		int[] danglingAlts = null;
		int numWarnings = 2;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testFollowReturnsToLoopReenteringSameRule() throws Exception {
		// D07 can be matched in the (...)? or fall out of esc back into (..)*
		// loop in sl.  Note that D07 is matched by ~(R|SLASH).  No good
		// way to write that grammar I guess
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"sl : L ( esc | ~(R|SLASH) )* R ;\n" +
			"\n" +
			"esc : SLASH ( N | D03 (D07)? ) ;");
		String expecting =
			".s0-R->:s3=>3\n" +
			".s0-SLASH->:s1=>1\n" +
			".s0-{L, N..D07}->:s2=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "D07";
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testTokenCallsAnotherOnLeftEdge() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"F   :   I '.'\n" +
			"    ;\n" +
			"I   :   '0'\n" +
			"    ;\n"
		);
		String expecting =
			".s0-'0'->.s1\n" +
			".s1-'.'->:s3=>1\n" +
			".s1-<EOT>->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}


	@Test public void testSelfRecursionAmbigAlts() throws Exception {
		// ambiguous grammar for "L ID R" (alts 1,2 of a)
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s : a;\n" +
			"a   :   L ID R\n" +
			"    |   L a R\n" + // disabled for L ID R
			"    |   b\n" +
			"    ;\n" +
			"\n" +
			"b   :   ID\n" +
			"    ;\n");
		String expecting =
			".s0-ID->:s5=>3\n" +
			".s0-L->.s1\n" +
			".s1-ID->.s2\n" +
			".s1-L->:s4=>2\n" +
			".s2-R->:s3=>1\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "L ID R";
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testIndirectRecursionAmbigAlts() throws Exception {
		// ambiguous grammar for "L ID R" (alts 1,2 of a)
		// This was derived from the java grammar 12/4/2004 when it
		// was not handling a unaryExpression properly.  I traced it
		// to incorrect closure-busy condition.  It thought that the trace
		// of a->b->a->b again for "L ID" was an infinite loop, but actually
		// the repeat call to b only happens *after* an L has been matched.
		// I added a check to see what the initial stack looks like and it
		// seems to work now.
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"s   :   a ;\n" +
			"a   :   L ID R\n" +
			"    |   b\n" +
			"    ;\n" +
			"\n" +
			"b   :   ID\n" +
			"    |   L a R\n" +
			"    ;");
		String expecting =
			".s0-ID->:s4=>2\n" +
			".s0-L->.s1\n" +
			".s1-ID->.s2\n" +
			".s1-L->:s4=>2\n" +
			".s2-R->:s3=>1\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "L ID R";
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testTailRecursionInvokedFromArbitraryLookaheadDecision() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : b X\n" +
			"  | b Y\n" +
			"  ;\n" +
			"\n" +
			"b : A\n" +
			"  | A b\n" +
			"  ;\n");
		List altsWithRecursion = Arrays.asList(new Object[] {1,2});
		assertNonLLStar(g, altsWithRecursion);
	}

	@Test public void testWildcardStarK1AndNonGreedyByDefaultInParser() throws Exception {
		// no error because .* assumes it should finish when it sees R
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"s : A block EOF ;\n" +
			"block : L .* R ;");
		String expecting =
			".s0-A..L->:s2=>1\n" +
			".s0-R->:s1=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testWildcardPlusK1AndNonGreedyByDefaultInParser() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"s : A block EOF ;\n" +
			"block : L .+ R ;");
		String expecting =
			".s0-A..L->:s2=>1\n" +
			".s0-R->:s1=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);
	}

	@Test public void testGatedSynPred() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"x   : (X)=> X\n" +
			"    | Y\n" +
			"    ;\n");
		String expecting =
			".s0-X&&{synpred1_t}?->:s1=>1\n" + // does not hoist; it gates edges
			".s0-Y->:s2=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);

		Set<String> preds = g.synPredNamesUsedInDFA;
		Set<String> expectedPreds = new HashSet<String>() {{add("synpred1_t");}};
		assertEquals("predicate names not recorded properly in grammar", expectedPreds, preds);
	}

	@Test public void testHoistedGatedSynPred() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"x   : (X)=> X\n" +
			"    | X\n" +
			"    ;\n");
		String expecting =
			".s0-X->.s1\n" +
			".s1-{synpred1_t}?->:s2=>1\n" + // hoists into decision
			".s1-{true}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, danglingAlts, numWarnings);

		Set<String> preds = g.synPredNamesUsedInDFA;
		Set<String> expectedPreds = new HashSet<String>() {{add("synpred1_t");}};
		assertEquals("predicate names not recorded properly in grammar", expectedPreds, preds);
	}

	// Check state table creation

	@Test public void testCyclicTableCreation() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A+ X | A+ Y ;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>2\n";
	}


	// S U P P O R T

	public void _template() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A | B;");
		String expecting =
			"\n";
		checkDecision(g, 1, expecting, null, null, null, null, 0);
	}

	protected void assertNonLLStar(Grammar g, List expectedBadAlts) {
		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// mimic actions of org.antlr.Tool first time for grammar g
		if ( g.getNumberOfDecisions()==0 ) {
			g.buildNFA();
			g.createLookaheadDFAs(false);
		}
		NonRegularDecisionMessage msg = getNonRegularDecisionMessage(equeue.errors);
		assertTrue("expected fatal non-LL(*) msg", msg!=null);
		List<Integer> alts = new ArrayList();
		alts.addAll(msg.altsWithRecursion);
		Collections.sort(alts);
		assertEquals(expectedBadAlts,alts);
	}

	protected void assertRecursionOverflow(Grammar g,
										   List expectedTargetRules,
										   int expectedAlt) {
		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// mimic actions of org.antlr.Tool first time for grammar g
		if ( g.getNumberOfDecisions()==0 ) {
			g.buildNFA();
			g.createLookaheadDFAs(false);
		}
		RecursionOverflowMessage msg = getRecursionOverflowMessage(equeue.errors);
		assertTrue("missing expected recursion overflow msg"+msg, msg!=null);
		assertEquals("target rules mismatch",
					 expectedTargetRules.toString(), msg.targetRules.toString());
		assertEquals("mismatched alt", expectedAlt, msg.alt);
	}

    @Test
    public void testWildcardInTreeGrammar() throws Exception {
        Grammar g = new Grammar(
            "tree grammar t;\n" +
            "a : A B | A . ;\n");
        String expecting =
            ".s0-A->.s1\n" +
            ".s1-A->:s3=>2\n" +
            ".s1-B->:s2=>1\n";
        int[] unreachableAlts = null;
        int[] nonDetAlts = new int[] {1,2};
        String ambigInput = null;
        int[] danglingAlts = null;
        int numWarnings = 1;
        checkDecision(g, 1, expecting, unreachableAlts,
                      nonDetAlts, ambigInput, danglingAlts, numWarnings);
    }

    @Test
    public void testWildcardInTreeGrammar2() throws Exception {
        Grammar g = new Grammar(
            "tree grammar t;\n" +
            "a : ^(A X Y) | ^(A . .) ;\n");
        String expecting =
            ".s0-A->.s1\n" +
            ".s1-DOWN->.s2\n" +
            ".s2-X->.s3\n" +
            ".s2-{A, Y}->:s6=>2\n" +
            ".s3-Y->.s4\n" +
            ".s3-{DOWN, A..X}->:s6=>2\n" +
            ".s4-DOWN->:s6=>2\n" +
            ".s4-UP->:s5=>1\n";
        int[] unreachableAlts = null;
        int[] nonDetAlts = new int[] {1,2};
        String ambigInput = null;
        int[] danglingAlts = null;
        int numWarnings = 1;
        checkDecision(g, 1, expecting, unreachableAlts,
                      nonDetAlts, ambigInput, danglingAlts, numWarnings);
    }

    protected void checkDecision(Grammar g,
								 int decision,
								 String expecting,
								 int[] expectingUnreachableAlts,
								 int[] expectingNonDetAlts,
								 String expectingAmbigInput,
								 int[] expectingDanglingAlts,
								 int expectingNumWarnings)
		throws Exception
	{
		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// mimic actions of org.antlr.Tool first time for grammar g
		if ( g.getNumberOfDecisions()==0 ) {
			g.buildNFA();
			g.createLookaheadDFAs(false);
		}
		CodeGenerator generator = new CodeGenerator(newTool(), g, "Java");
		g.setCodeGenerator(generator);

		if ( equeue.size()!=expectingNumWarnings ) {
			System.err.println("Warnings issued: "+equeue);
		}

		assertEquals("unexpected number of expected problems",
				   expectingNumWarnings, equeue.size());

		DFA dfa = g.getLookaheadDFA(decision);
		assertNotNull("no DFA for decision "+decision, dfa);
		FASerializer serializer = new FASerializer(g);
		String result = serializer.serialize(dfa.startState);

		List unreachableAlts = dfa.getUnreachableAlts();

		// make sure unreachable alts are as expected
		if ( expectingUnreachableAlts!=null ) {
			BitSet s = new BitSet();
			s.addAll(expectingUnreachableAlts);
			BitSet s2 = new BitSet();
			s2.addAll(unreachableAlts);
			assertEquals("unreachable alts mismatch", s, s2);
		}
		else {
			assertEquals("number of unreachable alts", 0,
						 unreachableAlts!=null?unreachableAlts.size():0);
		}

		// check conflicting input
		if ( expectingAmbigInput!=null ) {
			// first, find nondet message
			Message msg = (Message)equeue.warnings.get(0);
			assertTrue("expecting nondeterminism; found "+msg.getClass().getName(),
					    msg instanceof GrammarNonDeterminismMessage);
			GrammarNonDeterminismMessage nondetMsg =
				getNonDeterminismMessage(equeue.warnings);
			List labels =
				nondetMsg.probe.getSampleNonDeterministicInputSequence(nondetMsg.problemState);
			String input = nondetMsg.probe.getInputSequenceDisplay(labels);
			assertEquals(expectingAmbigInput, input);
		}

		// check nondet alts
		if ( expectingNonDetAlts!=null ) {
			RecursionOverflowMessage recMsg = null;
			GrammarNonDeterminismMessage nondetMsg =
				getNonDeterminismMessage(equeue.warnings);
			List nonDetAlts = null;
			if ( nondetMsg!=null ) {
				nonDetAlts =
					nondetMsg.probe.getNonDeterministicAltsForState(nondetMsg.problemState);
			}
			else {
				recMsg = getRecursionOverflowMessage(equeue.warnings);
				if ( recMsg!=null ) {
					//nonDetAlts = new ArrayList(recMsg.alts);
				}
			}
			// compare nonDetAlts with expectingNonDetAlts
			BitSet s = new BitSet();
			s.addAll(expectingNonDetAlts);
			BitSet s2 = new BitSet();
			s2.addAll(nonDetAlts);
			assertEquals("nondet alts mismatch", s, s2);
			assertTrue("found no nondet alts; expecting: "+
					    str(expectingNonDetAlts),
					    nondetMsg!=null||recMsg!=null);
		}
		else {
			// not expecting any nondet alts, make sure there are none
			GrammarNonDeterminismMessage nondetMsg =
				getNonDeterminismMessage(equeue.warnings);
			assertNull("found nondet alts, but expecting none", nondetMsg);
		}

		assertEquals(expecting, result);
	}

	protected GrammarNonDeterminismMessage getNonDeterminismMessage(List warnings) {
		for (int i = 0; i < warnings.size(); i++) {
			Message m = (Message) warnings.get(i);
			if ( m instanceof GrammarNonDeterminismMessage ) {
				return (GrammarNonDeterminismMessage)m;
			}
		}
		return null;
	}

	protected NonRegularDecisionMessage getNonRegularDecisionMessage(List errors) {
		for (int i = 0; i < errors.size(); i++) {
			Message m = (Message) errors.get(i);
			if ( m instanceof NonRegularDecisionMessage ) {
				return (NonRegularDecisionMessage)m;
			}
		}
		return null;
	}

	protected RecursionOverflowMessage getRecursionOverflowMessage(List warnings) {
		for (int i = 0; i < warnings.size(); i++) {
			Message m = (Message) warnings.get(i);
			if ( m instanceof RecursionOverflowMessage ) {
				return (RecursionOverflowMessage)m;
			}
		}
		return null;
	}

	protected LeftRecursionCyclesMessage getLeftRecursionCyclesMessage(List warnings) {
		for (int i = 0; i < warnings.size(); i++) {
			Message m = (Message) warnings.get(i);
			if ( m instanceof LeftRecursionCyclesMessage ) {
				return (LeftRecursionCyclesMessage)m;
			}
		}
		return null;
	}

	protected GrammarDanglingStateMessage getDanglingStateMessage(List warnings) {
		for (int i = 0; i < warnings.size(); i++) {
			Message m = (Message) warnings.get(i);
			if ( m instanceof GrammarDanglingStateMessage ) {
				return (GrammarDanglingStateMessage)m;
			}
		}
		return null;
	}

	protected String str(int[] elements) {
		StringBuffer buf = new StringBuffer();
		for (int i = 0; i < elements.length; i++) {
			if ( i>0 ) {
				buf.append(", ");
			}
			int element = elements[i];
			buf.append(element);
		}
		return buf.toString();
	}

	protected Set<String> ruleNames(Set<Rule> rules) {
		Set<String> x = new HashSet<String>();
		for (Rule r : rules) {
			x.add(r.name);
		}
		return x;
	}

	protected Set<String> ruleNames2(Collection<HashSet> rules) {
		Set<String> x = new HashSet<String>();
		for (HashSet s : rules) {
			x.addAll(ruleNames(s));
		}
		return x;
	}
}
