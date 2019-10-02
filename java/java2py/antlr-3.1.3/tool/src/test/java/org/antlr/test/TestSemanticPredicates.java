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
import org.antlr.codegen.CodeGenerator;
import org.antlr.misc.BitSet;
import org.antlr.tool.*;

import java.util.*;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

import antlr.Token;

public class TestSemanticPredicates extends BaseTest {

	/** Public default constructor used by TestRig */
	public TestSemanticPredicates() {
	}

	@Test public void testPredsButSyntaxResolves() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? A | {p2}? B ;");
		String expecting =
			".s0-A->:s1=>1\n" +
			".s0-B->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testLL_1_Pred() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? A | {p2}? A ;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{p2}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testLL_1_Pred_forced_k_1() throws Exception {
		// should stop just like before w/o k set.
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a options {k=1;} : {p1}? A | {p2}? A ;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{p2}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testLL_2_Pred() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? A B | {p2}? A B ;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->.s2\n" +
			".s2-{p1}?->:s3=>1\n" +
			".s2-{p2}?->:s4=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testPredicatedLoop() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : ( {p1}? A | {p2}? A )+;");
		String expecting =                   // loop back
			".s0-A->.s2\n" +
			".s0-EOF->:s1=>3\n" +
			".s2-{p1}?->:s3=>1\n" +
			".s2-{p2}?->:s4=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testPredicatedToStayInLoop() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : ( {p1}? A )+ (A)+;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-{!(p1)}?->:s2=>1\n" +
			".s1-{p1}?->:s3=>2\n";       // loop back
	}

	@Test public void testAndPredicates() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? {p1a}? A | {p2}? A ;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-{(p1&&p1a)}?->:s2=>1\n" +
			".s1-{p2}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test
    public void testOrPredicates() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b | {p2}? A ;\n" +
			"b : {p1}? A | {p1a}? A ;");
		String expecting =
			".s0-A->.s1\n" +
            ".s1-{(p1a||p1)}?->:s2=>1\n" +
            ".s1-{p2}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testIgnoresHoistingDepthGreaterThanZero() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : A {p1}? | A {p2}?;");
		String expecting =
			".s0-A->:s1=>1\n";
		checkDecision(g, 1, expecting, new int[] {2},
					  new int[] {1,2}, "A", null, null, 2, false);
	}

	@Test public void testIgnoresPredsHiddenByActions() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {a1} {p1}? A | {a2} {p2}? A ;");
		String expecting =
			".s0-A->:s1=>1\n";
		checkDecision(g, 1, expecting, new int[] {2},
					  new int[] {1,2}, "A", null, null, 2, true);
	}

	@Test public void testIgnoresPredsHiddenByActionsOneAlt() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? A | {a2} {p2}? A ;"); // ok since 1 pred visible
		String expecting =
			".s0-A->.s1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null,
					  null, null, null, null, 0, true);
	}

	/*
	@Test public void testIncompleteSemanticHoistedContextk2() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : b | A B;\n" +
			"b : {p1}? A B | A B ;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->:s2=>1\n";
		checkDecision(g, 1, expecting, new int[] {2},
					  new int[] {1,2}, "A B", new int[] {1}, null, 3);
	}	
	 */

	@Test public void testHoist2() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b | c ;\n" +
			"b : {p1}? A ;\n" +
			"c : {p2}? A ;\n");
		String expecting =
			".s0-A->.s1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{p2}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testHoistCorrectContext() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b | {p2}? ID ;\n" +
			"b : {p1}? ID | INT ;\n");
		String expecting =  // only tests after ID, not INT :)
			".s0-ID->.s1\n" +
			".s0-INT->:s2=>1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{p2}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testDefaultPredNakedAltIsLast() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b | ID ;\n" +
			"b : {p1}? ID | INT ;\n");
		String expecting =
			".s0-ID->.s1\n" +
			".s0-INT->:s2=>1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testDefaultPredNakedAltNotLast() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : ID | b ;\n" +
			"b : {p1}? ID | INT ;\n");
		String expecting =
			".s0-ID->.s1\n" +
			".s0-INT->:s3=>2\n" +
			".s1-{!(p1)}?->:s2=>1\n" +
			".s1-{p1}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testLeftRecursivePred() throws Exception {
		// No analysis possible. but probably good to fail.  Not sure we really want
		// left-recursion even if guarded with pred.
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"s : a ;\n" +
			"a : {p1}? a | ID ;\n");
		String expecting =
			".s0-ID->.s1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{true}?->:s3=>2\n";

		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		CodeGenerator generator = new CodeGenerator(newTool(), g, "Java");
		g.setCodeGenerator(generator);
		if ( g.getNumberOfDecisions()==0 ) {
			g.buildNFA();
			g.createLookaheadDFAs(false);
		}

		DFA dfa = g.getLookaheadDFA(1);
		assertEquals(null, dfa); // can't analyze.

		/*
		String result = serializer.serialize(dfa.startState);
		assertEquals(expecting, result);
		*/

		assertEquals("unexpected number of expected problems", 1, equeue.size());
		Message msg = (Message)equeue.warnings.get(0);
		assertTrue("warning must be a left recursion msg",
				    msg instanceof LeftRecursionCyclesMessage);
	}

	@Test public void testIgnorePredFromLL2AltLastAltIsDefaultTrue() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? A B | A C | {p2}? A | {p3}? A | A ;\n");
		// two situations of note:
		// 1. A B syntax is enough to predict that alt, so p1 is not used
		//    to distinguish it from alts 2..5
		// 2. Alts 3, 4, 5 are nondeterministic with upon A.  p2, p3 and the
		//    complement of p2||p3 is sufficient to resolve the conflict. Do
		//    not include alt 1's p1 pred in the "complement of other alts"
		//    because it is not considered nondeterministic with alts 3..5
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->:s2=>1\n" +
			".s1-C->:s3=>2\n" +
			".s1-{p2}?->:s4=>3\n" +
			".s1-{p3}?->:s5=>4\n" +
			".s1-{true}?->:s6=>5\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testIgnorePredFromLL2AltPredUnionNeeded() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : {p1}? A B | A C | {p2}? A | A | {p3}? A ;\n");
		// two situations of note:
		// 1. A B syntax is enough to predict that alt, so p1 is not used
		//    to distinguish it from alts 2..5
		// 2. Alts 3, 4, 5 are nondeterministic with upon A.  p2, p3 and the
		//    complement of p2||p3 is sufficient to resolve the conflict. Do
		//    not include alt 1's p1 pred in the "complement of other alts"
		//    because it is not considered nondeterministic with alts 3..5
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->:s2=>1\n" +
			".s1-C->:s3=>2\n" +
			".s1-{!((p3||p2))}?->:s5=>4\n" +
			".s1-{p2}?->:s4=>3\n" +
			".s1-{p3}?->:s6=>5\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testPredGets2SymbolSyntacticContext() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b | A B | C ;\n" +
			"b : {p1}? A B ;\n");
		String expecting =
			".s0-A->.s1\n" +
			".s0-C->:s5=>3\n" +
			".s1-B->.s2\n" +
			".s2-{p1}?->:s3=>1\n" +
			".s2-{true}?->:s4=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testMatchesLongestThenTestPred() throws Exception {
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"a : b | c ;\n" +
			"b : {p}? A ;\n" +
			"c : {q}? (A|B)+ ;");
		String expecting =
			".s0-A->.s1\n" +
			".s0-B->:s3=>2\n" +
			".s1-{p}?->:s2=>1\n" +
			".s1-{q}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testPredsUsedAfterRecursionOverflow() throws Exception {
		// analysis must bail out due to non-LL(*) nature (ovf)
		// retries with k=1 (but with LL(*) algorithm not optimized version
		// as it has preds)
		Grammar g = new Grammar(
			"parser grammar P;\n"+
			"s : {p1}? e '.' | {p2}? e ':' ;\n" +
			"e : '(' e ')' | INT ;\n");
		String expecting =
			".s0-'('->.s1\n" +
			".s0-INT->.s4\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{p2}?->:s3=>2\n" +
			".s4-{p1}?->:s2=>1\n" +
			".s4-{p2}?->:s3=>2\n";
		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		CodeGenerator generator = new CodeGenerator(newTool(), g, "Java");
		g.setCodeGenerator(generator);
		if ( g.getNumberOfDecisions()==0 ) {
			g.buildNFA();
			g.createLookaheadDFAs(false);
		}

		assertEquals("unexpected number of expected problems", 0, equeue.size());
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testPredsUsedAfterK2FailsNoRecursionOverflow() throws Exception {
		// analysis must bail out due to non-LL(*) nature (ovf)
		// retries with k=1 (but with LL(*) algorithm not optimized version
		// as it has preds)
		Grammar g = new Grammar(
			"grammar P;\n" +
			"options {k=2;}\n"+
			"s : {p1}? e '.' | {p2}? e ':' ;\n" +
			"e : '(' e ')' | INT ;\n");
		String expecting =
			".s0-'('->.s1\n" +
			".s0-INT->.s6\n" +
			".s1-'('->.s2\n" +
			".s1-INT->.s5\n" +
			".s2-{p1}?->:s3=>1\n" +
			".s2-{p2}?->:s4=>2\n" +
			".s5-{p1}?->:s3=>1\n" +
			".s5-{p2}?->:s4=>2\n" +
			".s6-'.'->:s3=>1\n" +
			".s6-':'->:s4=>2\n";
		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		CodeGenerator generator = new CodeGenerator(newTool(), g, "Java");
		g.setCodeGenerator(generator);
		if ( g.getNumberOfDecisions()==0 ) {
			g.buildNFA();
			g.createLookaheadDFAs(false);
		}

		assertEquals("unexpected number of expected problems", 0, equeue.size());
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testLexerMatchesLongestThenTestPred() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"B : {p}? 'a' ;\n" +
			"C : {q}? ('a'|'b')+ ;");
		String expecting =
			".s0-'a'->.s1\n" +
			".s0-'b'->:s4=>2\n" +
			".s1-'a'..'b'->:s4=>2\n" +
			".s1-<EOT>->.s2\n" +
			".s2-{p}?->:s3=>1\n" +
			".s2-{q}?->:s4=>2\n";
		checkDecision(g, 2, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testLexerMatchesLongestMinusPred() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"B : 'a' ;\n" +
			"C : ('a'|'b')+ ;");
		String expecting =
			".s0-'a'->.s1\n" +
			".s0-'b'->:s3=>2\n" +
			".s1-'a'..'b'->:s3=>2\n" +
			".s1-<EOT>->:s2=>1\n";
		checkDecision(g, 2, expecting, null, null, null, null, null, 0, false);
	}

    @Test
    public void testGatedPred() throws Exception {
		// gated preds are present on all arcs in predictor
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"B : {p}? => 'a' ;\n" +
			"C : {q}? => ('a'|'b')+ ;");
		String expecting =
			".s0-'a'&&{(q||p)}?->.s1\n" +
            ".s0-'b'&&{q}?->:s4=>2\n" +
            ".s1-'a'..'b'&&{q}?->:s4=>2\n" +
            ".s1-<EOT>&&{(q||p)}?->.s2\n" +
            ".s2-{p}?->:s3=>1\n" +
            ".s2-{q}?->:s4=>2\n";
		checkDecision(g, 2, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testGatedPredHoistsAndCanBeInStopState() throws Exception {
		// I found a bug where merging stop states made us throw away
		// a stop state with a gated pred!
		Grammar g = new Grammar(
			"grammar u;\n" +
			"a : b+ ;\n" +
			"b : 'x' | {p}?=> 'y' ;");
		String expecting =
			".s0-'x'->:s2=>1\n" +
			".s0-'y'&&{p}?->:s3=>1\n" +
			".s0-EOF->:s1=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test
    public void testGatedPredInCyclicDFA() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : {p}?=> ('a')+ 'x' ;\n" +
			"B : {q}?=> ('a'|'b')+ 'x' ;");
		String expecting =
			".s0-'a'&&{(q||p)}?->.s1\n" +
            ".s0-'b'&&{q}?->:s5=>2\n" +
            ".s1-'a'&&{(q||p)}?->.s1\n" +
            ".s1-'b'&&{q}?->:s5=>2\n" +
            ".s1-'x'&&{(q||p)}?->.s2\n" +
            ".s2-<EOT>&&{(q||p)}?->.s3\n" +
            ".s3-{p}?->:s4=>1\n" +
            ".s3-{q}?->:s5=>2\n";
		checkDecision(g, 3, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testGatedPredNotActuallyUsedOnEdges() throws Exception {
		Grammar g = new Grammar(
			"lexer grammar P;\n"+
			"A : ('a' | {p}?=> 'a')\n" +
			"  | 'a' 'b'\n" +
			"  ;");
		String expecting1 =
			".s0-'a'->.s1\n" +
			".s1-{!(p)}?->:s2=>1\n" +  	// Used to disambig subrule
			".s1-{p}?->:s3=>2\n";
		// rule A decision can't test p from s0->1 because 'a' is valid
		// for alt1 *and* alt2 w/o p.  Can't test p from s1 to s3 because
		// we might have passed the first alt of subrule.  The same state
		// is listed in s2 in 2 different configurations: one with and one
		// w/o p.  Can't test therefore.  p||true == true.
		String expecting2 =
			".s0-'a'->.s1\n" +
			".s1-'b'->:s2=>2\n" +
			".s1-<EOT>->:s3=>1\n";
		checkDecision(g, 1, expecting1, null, null, null, null, null, 0, false);
		checkDecision(g, 2, expecting2, null, null, null, null, null, 0, false);
	}

	@Test public void testGatedPredDoesNotForceAllToBeGated() throws Exception {
		Grammar g = new Grammar(
			"grammar w;\n" +
			"a : b | c ;\n" +
			"b : {p}? B ;\n" +
			"c : {q}?=> d ;\n" +
			"d : {r}? C ;\n");
		String expecting =
			".s0-B->:s1=>1\n" +
			".s0-C&&{q}?->:s2=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testGatedPredDoesNotForceAllToBeGated2() throws Exception {
		Grammar g = new Grammar(
			"grammar w;\n" +
			"a : b | c ;\n" +
			"b : {p}? B ;\n" +
			"c : {q}?=> d ;\n" +
			"d : {r}?=> C\n" +
			"  | B\n" +
			"  ;\n");
		String expecting =
			".s0-B->.s1\n" +
			".s0-C&&{(q&&r)}?->:s3=>2\n" +
			".s1-{p}?->:s2=>1\n" +
			".s1-{q}?->:s3=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	@Test public void testORGatedPred() throws Exception {
		Grammar g = new Grammar(
			"grammar w;\n" +
			"a : b | c ;\n" +
			"b : {p}? B ;\n" +
			"c : {q}?=> d ;\n" +
			"d : {r}?=> C\n" +
			"  | {s}?=> B\n" +
			"  ;\n");
		String expecting =
			".s0-B->.s1\n" +
			".s0-C&&{(q&&r)}?->:s3=>2\n" +
			".s1-{(q&&s)}?->:s3=>2\n" +
			".s1-{p}?->:s2=>1\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	/** The following grammar should yield an error that rule 'a' has
	 *  insufficient semantic info pulled from 'b'.
	 */
	@Test public void testIncompleteSemanticHoistedContext() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : b | B;\n" +
			"b : {p1}? B | B ;");
		String expecting =
			".s0-B->:s1=>1\n";
		checkDecision(g, 1, expecting, new int[] {2},
					  new int[] {1,2}, "B", new int[] {1}, null, 3, false);
	}

	@Test public void testIncompleteSemanticHoistedContextk2() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : b | A B;\n" +
			"b : {p1}? A B | A B ;");
		String expecting =
			".s0-A->.s1\n" +
			".s1-B->:s2=>1\n";
		checkDecision(g, 1, expecting, new int[] {2},
					  new int[] {1,2}, "A B", new int[] {1}, null, 3, false);
	}

	@Test public void testIncompleteSemanticHoistedContextInFOLLOW() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"options {k=1;}\n" + // limit to k=1 because it's LL(2); force pred hoist
			"a : A? ;\n" + // need FOLLOW
			"b : X a {p1}? A | Y a A ;"); // only one A is covered
		String expecting =
			".s0-A->:s1=>1\n"; // s0-EOF->s2 branch pruned during optimization
		checkDecision(g, 1, expecting, new int[] {2},
					  new int[] {1,2}, "A", new int[] {2}, null, 3, false);
	}

	@Test public void testIncompleteSemanticHoistedContextInFOLLOWk2() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A B)? ;\n" + // need FOLLOW
			"b : X a {p1}? A B | Y a A B | Z a ;"); // only first alt is covered
		String expecting =
			".s0-A->.s1\n" +
			".s0-EOF->:s3=>2\n" +
			".s1-B->:s2=>1\n";
		checkDecision(g, 1, expecting, null,
					  new int[] {1,2}, "A B", new int[] {2}, null, 2, false);
	}

	@Test public void testIncompleteSemanticHoistedContextInFOLLOWDueToHiddenPred() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : (A B)? ;\n" + // need FOLLOW
			"b : X a {p1}? A B | Y a {a1} {p2}? A B | Z a ;"); // only first alt is covered
		String expecting =
			".s0-A->.s1\n" +
			".s0-EOF->:s3=>2\n" +
			".s1-B->:s2=>1\n";
		checkDecision(g, 1, expecting, null,
					  new int[] {1,2}, "A B", new int[] {2}, null, 2, true);
	}

	/** The following grammar should yield an error that rule 'a' has
	 *  insufficient semantic info pulled from 'b'.  This is the same
	 *  as the previous case except that the D prevents the B path from
	 *  "pinching" together into a single NFA state.
	 *
	 *  This test also demonstrates that just because B D could predict
	 *  alt 1 in rule 'a', it is unnecessary to continue NFA->DFA
	 *  conversion to include an edge for D.  Alt 1 is the only possible
	 *  prediction because we resolve the ambiguity by choosing alt 1.
	 */
	@Test public void testIncompleteSemanticHoistedContext2() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : b | B;\n" +
			"b : {p1}? B | B D ;");
		String expecting =
			".s0-B->:s1=>1\n";
		checkDecision(g, 1, expecting, new int[] {2},
					  new int[] {1,2}, "B", new int[] {1},
					  null, 3, false);
	}

	@Test public void testTooFewSemanticPredicates() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : {p1}? A | A | A ;");
		String expecting =
			".s0-A->:s1=>1\n";
		checkDecision(g, 1, expecting, new int[] {2,3},
					  new int[] {1,2,3}, "A",
					  null, null, 2, false);
	}

	@Test public void testPredWithK1() throws Exception {
		Grammar g = new Grammar(
			"\tlexer grammar TLexer;\n" +
			"A\n" +
			"options {\n" +
			"  k=1;\n" +
			"}\n" +
			"  : {p1}? ('x')+ '.'\n" +
			"  | {p2}? ('x')+ '.'\n" +
			"  ;\n");
		String expecting =
			".s0-'x'->.s1\n" +
			".s1-{p1}?->:s2=>1\n" +
			".s1-{p2}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] insufficientPredAlts = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 3, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, insufficientPredAlts,
					  danglingAlts, numWarnings, false);
	}

	@Test public void testPredWithArbitraryLookahead() throws Exception {
		Grammar g = new Grammar(
			"\tlexer grammar TLexer;\n" +
			"A : {p1}? ('x')+ '.'\n" +
			"  | {p2}? ('x')+ '.'\n" +
			"  ;\n");
		String expecting =
			".s0-'x'->.s1\n" +
			".s1-'.'->.s2\n" +
			".s1-'x'->.s1\n" +
			".s2-{p1}?->:s3=>1\n" +
			".s2-{p2}?->:s4=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] insufficientPredAlts = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 3, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, insufficientPredAlts,
					  danglingAlts, numWarnings, false);
	}

	@Test
    /** For a DFA state with lots of configurations that have the same
	 *  predicate, don't just OR them all together as it's a waste to
	 *  test a||a||b||a||a etc...  ANTLR makes a unique set and THEN
	 *  OR's them together.
	 */
    public void testUniquePredicateOR() throws Exception {
		Grammar g = new Grammar(
			"parser grammar v;\n" +
			"\n" +
			"a : {a}? b\n" +
			"  | {b}? b\n" +
			"  ;\n" +
			"\n" +
			"b : {c}? (X)+ ;\n" +
			"\n" +
			"c : a\n" +
			"  | b\n" +
			"  ;\n");
		String expecting =
			".s0-X->.s1\n" +
            ".s1-{((a&&c)||(b&&c))}?->:s2=>1\n" +
            ".s1-{c}?->:s3=>2\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = null;
		String ambigInput = null;
		int[] insufficientPredAlts = null;
		int[] danglingAlts = null;
		int numWarnings = 0;
		checkDecision(g, 3, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, insufficientPredAlts,
					  danglingAlts, numWarnings, false);
	}

    @Test
    public void testSemanticContextPreventsEarlyTerminationOfClosure() throws Exception {
		Grammar g = new Grammar(
			"parser grammar T;\n" +
			"a : loop SEMI | ID SEMI\n" +
			"  ;\n" +
			"loop\n" +
			"    : {while}? ID\n" +
			"    | {do}? ID\n" +
			"    | {for}? ID\n" +
			"    ;");
		String expecting =
			".s0-ID->.s1\n" +
            ".s1-SEMI->.s2\n" +
            ".s2-{(for||do||while)}?->:s3=>1\n" +
            ".s2-{true}?->:s4=>2\n";
		checkDecision(g, 1, expecting, null, null, null, null, null, 0, false);
	}

	// S U P P O R T

	public void _template() throws Exception {
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A | B;");
		String expecting =
			"\n";
		int[] unreachableAlts = null;
		int[] nonDetAlts = new int[] {1,2};
		String ambigInput = "L ID R";
		int[] insufficientPredAlts = new int[] {1};
		int[] danglingAlts = null;
		int numWarnings = 1;
		checkDecision(g, 1, expecting, unreachableAlts,
					  nonDetAlts, ambigInput, insufficientPredAlts,
					  danglingAlts, numWarnings, false);
	}

	protected void checkDecision(Grammar g,
								 int decision,
								 String expecting,
								 int[] expectingUnreachableAlts,
								 int[] expectingNonDetAlts,
								 String expectingAmbigInput,
								 int[] expectingInsufficientPredAlts,
								 int[] expectingDanglingAlts,
								 int expectingNumWarnings,
								 boolean hasPredHiddenByAction)
		throws Exception
	{
		DecisionProbe.verbose=true; // make sure we get all error info
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		CodeGenerator generator = new CodeGenerator(newTool(), g, "Java");
		g.setCodeGenerator(generator);
		// mimic actions of org.antlr.Tool first time for grammar g
		if ( g.getNumberOfDecisions()==0 ) {
			g.buildNFA();
			g.createLookaheadDFAs(false);
		}

		if ( equeue.size()!=expectingNumWarnings ) {
			System.err.println("Warnings issued: "+equeue);
		}

		assertEquals("unexpected number of expected problems",
				   expectingNumWarnings, equeue.size());

		DFA dfa = g.getLookaheadDFA(decision);
		FASerializer serializer = new FASerializer(g);
		String result = serializer.serialize(dfa.startState);
		//System.out.print(result);
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
			assertEquals("unreachable alts mismatch", 0,
						 unreachableAlts!=null?unreachableAlts.size():0);
		}

		// check conflicting input
		if ( expectingAmbigInput!=null ) {
			// first, find nondet message
			Message msg = getNonDeterminismMessage(equeue.warnings);
			assertNotNull("no nondeterminism warning?", msg);
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
			GrammarNonDeterminismMessage nondetMsg =
				getNonDeterminismMessage(equeue.warnings);
			assertNotNull("found no nondet alts; expecting: "+
										str(expectingNonDetAlts), nondetMsg);
			List nonDetAlts =
				nondetMsg.probe.getNonDeterministicAltsForState(nondetMsg.problemState);
			// compare nonDetAlts with expectingNonDetAlts
			BitSet s = new BitSet();
			s.addAll(expectingNonDetAlts);
			BitSet s2 = new BitSet();
			s2.addAll(nonDetAlts);
			assertEquals("nondet alts mismatch", s, s2);
			assertEquals("mismatch between expected hasPredHiddenByAction", hasPredHiddenByAction,
						 nondetMsg.problemState.dfa.hasPredicateBlockedByAction);
		}
		else {
			// not expecting any nondet alts, make sure there are none
			GrammarNonDeterminismMessage nondetMsg =
				getNonDeterminismMessage(equeue.warnings);
			assertNull("found nondet alts, but expecting none", nondetMsg);
		}

		if ( expectingInsufficientPredAlts!=null ) {
			GrammarInsufficientPredicatesMessage insuffPredMsg =
				getGrammarInsufficientPredicatesMessage(equeue.warnings);
			assertNotNull("found no GrammarInsufficientPredicatesMessage alts; expecting: "+
										str(expectingNonDetAlts), insuffPredMsg);
			Map<Integer, Set<Token>> locations = insuffPredMsg.altToLocations;
			Set actualAlts = locations.keySet();
			BitSet s = new BitSet();
			s.addAll(expectingInsufficientPredAlts);
			BitSet s2 = new BitSet();
			s2.addAll(actualAlts);
			assertEquals("mismatch between insufficiently covered alts", s, s2);
			assertEquals("mismatch between expected hasPredHiddenByAction", hasPredHiddenByAction,
						 insuffPredMsg.problemState.dfa.hasPredicateBlockedByAction);
		}
		else {
			// not expecting any nondet alts, make sure there are none
			GrammarInsufficientPredicatesMessage nondetMsg =
				getGrammarInsufficientPredicatesMessage(equeue.warnings);
			if ( nondetMsg!=null ) {
				System.out.println(equeue.warnings);
			}
			assertNull("found insufficiently covered alts, but expecting none", nondetMsg);
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

	protected GrammarInsufficientPredicatesMessage getGrammarInsufficientPredicatesMessage(List warnings) {
		for (int i = 0; i < warnings.size(); i++) {
			Message m = (Message) warnings.get(i);
			if ( m instanceof GrammarInsufficientPredicatesMessage ) {
				return (GrammarInsufficientPredicatesMessage)m;
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
}
