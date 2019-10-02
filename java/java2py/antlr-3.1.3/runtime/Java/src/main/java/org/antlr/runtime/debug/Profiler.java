/*
 [The "BSD licence"]
 Copyright (c) 2005-2008 Terence Parr
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
package org.antlr.runtime.debug;

import org.antlr.runtime.*;
import org.antlr.runtime.misc.Stats;

import java.util.*;
import java.io.IOException;

/** Using the debug event interface, track what is happening in the parser
 *  and record statistics about the runtime.
 */
public class Profiler extends BlankDebugEventListener {
	/** Because I may change the stats, I need to track that for later
	 *  computations to be consistent.
	 */
	public static final String Version = "2";
	public static final String RUNTIME_STATS_FILENAME = "runtime.stats";
	public static final int NUM_RUNTIME_STATS = 29;

	public DebugParser parser = null;

	// working variables

	protected int ruleLevel = 0;
	protected int decisionLevel = 0;
	protected int maxLookaheadInCurrentDecision = 0;
	protected CommonToken lastTokenConsumed=null;

	protected List lookaheadStack = new ArrayList();

	// stats variables

	public int numRuleInvocations = 0;
	public int numGuessingRuleInvocations = 0;
	public int maxRuleInvocationDepth = 0;
	public int numFixedDecisions = 0;
	public int numCyclicDecisions = 0;
	public int numBacktrackDecisions = 0;
	public int[] decisionMaxFixedLookaheads = new int[200]; // TODO: make List
	public int[] decisionMaxCyclicLookaheads = new int[200];
	public List decisionMaxSynPredLookaheads = new ArrayList();
	public int numHiddenTokens = 0;
	public int numCharsMatched = 0;
	public int numHiddenCharsMatched = 0;
	public int numSemanticPredicates = 0;
	public int numSyntacticPredicates = 0;
	protected int numberReportedErrors = 0;
	public int numMemoizationCacheMisses = 0;
	public int numMemoizationCacheHits = 0;
	public int numMemoizationCacheEntries = 0;

	public Profiler() {
	}

	public Profiler(DebugParser parser) {
		this.parser = parser;
	}

	public void enterRule(String grammarFileName, String ruleName) {
		//System.out.println("enterRule "+ruleName);
		ruleLevel++;
		numRuleInvocations++;
		if ( ruleLevel >maxRuleInvocationDepth ) {
			maxRuleInvocationDepth = ruleLevel;
		}

	}

	/** Track memoization; this is not part of standard debug interface
	 *  but is triggered by profiling.  Code gen inserts an override
	 *  for this method in the recognizer, which triggers this method.
	 */
	public void examineRuleMemoization(IntStream input,
									   int ruleIndex,
									   String ruleName)
	{
		//System.out.println("examine memo "+ruleName);
		int stopIndex = parser.getRuleMemoization(ruleIndex, input.index());
		if ( stopIndex==BaseRecognizer.MEMO_RULE_UNKNOWN ) {
			//System.out.println("rule "+ruleIndex+" missed @ "+input.index());
			numMemoizationCacheMisses++;
			numGuessingRuleInvocations++; // we'll have to enter
		}
		else {
			// regardless of rule success/failure, if in cache, we have a cache hit
			//System.out.println("rule "+ruleIndex+" hit @ "+input.index());
			numMemoizationCacheHits++;
		}
	}

	public void memoize(IntStream input,
						int ruleIndex,
						int ruleStartIndex,
						String ruleName)
	{
		// count how many entries go into table
		//System.out.println("memoize "+ruleName);
		numMemoizationCacheEntries++;
	}

	public void exitRule(String grammarFileName, String ruleName) {
		ruleLevel--;
	}

	public void enterDecision(int decisionNumber) {
		decisionLevel++;
		int startingLookaheadIndex = parser.getTokenStream().index();
		//System.out.println("enterDecision "+decisionNumber+" @ index "+startingLookaheadIndex);
		lookaheadStack.add(new Integer(startingLookaheadIndex));
	}

	public void exitDecision(int decisionNumber) {
		//System.out.println("exitDecision "+decisionNumber);
		// track how many of acyclic, cyclic here as we don't know what kind
		// yet in enterDecision event.
		if ( parser.isCyclicDecision ) {
			numCyclicDecisions++;
		}
		else {
			numFixedDecisions++;
		}
		lookaheadStack.remove(lookaheadStack.size()-1); // pop lookahead depth counter
		decisionLevel--;
		if ( parser.isCyclicDecision ) {
			if ( numCyclicDecisions>=decisionMaxCyclicLookaheads.length ) {
				int[] bigger = new int[decisionMaxCyclicLookaheads.length*2];
				System.arraycopy(decisionMaxCyclicLookaheads,0,bigger,0,decisionMaxCyclicLookaheads.length);
				decisionMaxCyclicLookaheads = bigger;
			}
			decisionMaxCyclicLookaheads[numCyclicDecisions-1] = maxLookaheadInCurrentDecision;
		}
		else {
			if ( numFixedDecisions>=decisionMaxFixedLookaheads.length ) {
				int[] bigger = new int[decisionMaxFixedLookaheads.length*2];
				System.arraycopy(decisionMaxFixedLookaheads,0,bigger,0,decisionMaxFixedLookaheads.length);
				decisionMaxFixedLookaheads = bigger;
			}
			decisionMaxFixedLookaheads[numFixedDecisions-1] = maxLookaheadInCurrentDecision;
		}
		parser.isCyclicDecision = false; // can't nest so just reset to false
		maxLookaheadInCurrentDecision = 0;
	}

	public void consumeToken(Token token) {
		//System.out.println("consume token "+token);
		lastTokenConsumed = (CommonToken)token;
	}

	/** The parser is in a decision if the decision depth > 0.  This
	 *  works for backtracking also, which can have nested decisions.
	 */
	public boolean inDecision() {
		return decisionLevel>0;
	}

	public void consumeHiddenToken(Token token) {
		//System.out.println("consume hidden token "+token);
		lastTokenConsumed = (CommonToken)token;
	}

	/** Track refs to lookahead if in a fixed/nonfixed decision.
	 */
	public void LT(int i, Token t) {
		if ( inDecision() ) {
			// get starting index off stack
			int stackTop = lookaheadStack.size()-1;
			Integer startingIndex = (Integer)lookaheadStack.get(stackTop);
			// compute lookahead depth
			int thisRefIndex = parser.getTokenStream().index();
			int numHidden =
				getNumberOfHiddenTokens(startingIndex.intValue(), thisRefIndex);
			int depth = i + thisRefIndex - startingIndex.intValue() - numHidden;
			/*
			System.out.println("LT("+i+") @ index "+thisRefIndex+" is depth "+depth+
				" max is "+maxLookaheadInCurrentDecision);
			*/
			if ( depth>maxLookaheadInCurrentDecision ) {
				maxLookaheadInCurrentDecision = depth;
			}
		}
	}

	/** Track backtracking decisions.  You'll see a fixed or cyclic decision
	 *  and then a backtrack.
	 *
	 * 		enter rule
	 * 		...
	 * 		enter decision
	 * 		LA and possibly consumes (for cyclic DFAs)
	 * 		begin backtrack level
	 * 		mark m
	 * 		rewind m
	 * 		end backtrack level, success
	 * 		exit decision
	 * 		...
	 * 		exit rule
	 */
	public void beginBacktrack(int level) {
		//System.out.println("enter backtrack "+level);
		numBacktrackDecisions++;
	}

	/** Successful or not, track how much lookahead synpreds use */
	public void endBacktrack(int level, boolean successful) {
		//System.out.println("exit backtrack "+level+": "+successful);
		decisionMaxSynPredLookaheads.add(
			new Integer(maxLookaheadInCurrentDecision)
		);
	}

	/*
	public void mark(int marker) {
		int i = parser.getTokenStream().index();
		System.out.println("mark @ index "+i);
		synPredLookaheadStack.add(new Integer(i));
	}

	public void rewind(int marker) {
		// pop starting index off stack
		int stackTop = synPredLookaheadStack.size()-1;
		Integer startingIndex = (Integer)synPredLookaheadStack.get(stackTop);
		synPredLookaheadStack.remove(synPredLookaheadStack.size()-1);
		// compute lookahead depth
		int stopIndex = parser.getTokenStream().index();
		System.out.println("rewind @ index "+stopIndex);
		int depth = stopIndex - startingIndex.intValue();
		System.out.println("depth of lookahead for synpred: "+depth);
		decisionMaxSynPredLookaheads.add(
			new Integer(depth)
		);
	}
	*/

	public void recognitionException(RecognitionException e) {
		numberReportedErrors++;
	}

	public void semanticPredicate(boolean result, String predicate) {
		if ( inDecision() ) {
			numSemanticPredicates++;
		}
	}

	public void terminate() {
		String stats = toNotifyString();
		try {
			Stats.writeReport(RUNTIME_STATS_FILENAME,stats);
		}
		catch (IOException ioe) {
			System.err.println(ioe);
			ioe.printStackTrace(System.err);
		}
		System.out.println(toString(stats));
	}

	public void setParser(DebugParser parser) {
		this.parser = parser;
	}

	// R E P O R T I N G

	public String toNotifyString() {
		TokenStream input = parser.getTokenStream();
		for (int i=0; i<input.size()&&lastTokenConsumed!=null&&i<=lastTokenConsumed.getTokenIndex(); i++) {
			Token t = input.get(i);
			if ( t.getChannel()!=Token.DEFAULT_CHANNEL ) {
				numHiddenTokens++;
				numHiddenCharsMatched += t.getText().length();
			}
		}
		numCharsMatched = lastTokenConsumed.getStopIndex() + 1;
		decisionMaxFixedLookaheads = trim(decisionMaxFixedLookaheads, numFixedDecisions);
		decisionMaxCyclicLookaheads = trim(decisionMaxCyclicLookaheads, numCyclicDecisions);
		StringBuffer buf = new StringBuffer();
		buf.append(Version);
		buf.append('\t');
		buf.append(parser.getClass().getName());
		buf.append('\t');
		buf.append(numRuleInvocations);
		buf.append('\t');
		buf.append(maxRuleInvocationDepth);
		buf.append('\t');
		buf.append(numFixedDecisions);
		buf.append('\t');
		buf.append(Stats.min(decisionMaxFixedLookaheads));
		buf.append('\t');
		buf.append(Stats.max(decisionMaxFixedLookaheads));
		buf.append('\t');
		buf.append(Stats.avg(decisionMaxFixedLookaheads));
		buf.append('\t');
		buf.append(Stats.stddev(decisionMaxFixedLookaheads));
		buf.append('\t');
		buf.append(numCyclicDecisions);
		buf.append('\t');
		buf.append(Stats.min(decisionMaxCyclicLookaheads));
		buf.append('\t');
		buf.append(Stats.max(decisionMaxCyclicLookaheads));
		buf.append('\t');
		buf.append(Stats.avg(decisionMaxCyclicLookaheads));
		buf.append('\t');
		buf.append(Stats.stddev(decisionMaxCyclicLookaheads));
		buf.append('\t');
		buf.append(numBacktrackDecisions);
		buf.append('\t');
		buf.append(Stats.min(toArray(decisionMaxSynPredLookaheads)));
		buf.append('\t');
		buf.append(Stats.max(toArray(decisionMaxSynPredLookaheads)));
		buf.append('\t');
		buf.append(Stats.avg(toArray(decisionMaxSynPredLookaheads)));
		buf.append('\t');
		buf.append(Stats.stddev(toArray(decisionMaxSynPredLookaheads)));
		buf.append('\t');
		buf.append(numSemanticPredicates);
		buf.append('\t');
		buf.append(parser.getTokenStream().size());
		buf.append('\t');
		buf.append(numHiddenTokens);
		buf.append('\t');
		buf.append(numCharsMatched);
		buf.append('\t');
		buf.append(numHiddenCharsMatched);
		buf.append('\t');
		buf.append(numberReportedErrors);
		buf.append('\t');
		buf.append(numMemoizationCacheHits);
		buf.append('\t');
		buf.append(numMemoizationCacheMisses);
		buf.append('\t');
		buf.append(numGuessingRuleInvocations);
		buf.append('\t');
		buf.append(numMemoizationCacheEntries);
		return buf.toString();
	}

	public String toString() {
		return toString(toNotifyString());
	}

	protected static String[] decodeReportData(String data) {
		String[] fields = new String[NUM_RUNTIME_STATS];
		StringTokenizer st = new StringTokenizer(data, "\t");
		int i = 0;
		while ( st.hasMoreTokens() ) {
			fields[i] = st.nextToken();
			i++;
		}
		if ( i!=NUM_RUNTIME_STATS ) {
			return null;
		}
		return fields;
	}

	public static String toString(String notifyDataLine) {
		String[] fields = decodeReportData(notifyDataLine);
		if ( fields==null ) {
			return null;
		}
		StringBuffer buf = new StringBuffer();
		buf.append("ANTLR Runtime Report; Profile Version ");
		buf.append(fields[0]);
		buf.append('\n');
		buf.append("parser name ");
		buf.append(fields[1]);
		buf.append('\n');
		buf.append("Number of rule invocations ");
		buf.append(fields[2]);
		buf.append('\n');
		buf.append("Number of rule invocations in \"guessing\" mode ");
		buf.append(fields[27]);
		buf.append('\n');
		buf.append("max rule invocation nesting depth ");
		buf.append(fields[3]);
		buf.append('\n');
		buf.append("number of fixed lookahead decisions ");
		buf.append(fields[4]);
		buf.append('\n');
		buf.append("min lookahead used in a fixed lookahead decision ");
		buf.append(fields[5]);
		buf.append('\n');
		buf.append("max lookahead used in a fixed lookahead decision ");
		buf.append(fields[6]);
		buf.append('\n');
		buf.append("average lookahead depth used in fixed lookahead decisions ");
		buf.append(fields[7]);
		buf.append('\n');
		buf.append("standard deviation of depth used in fixed lookahead decisions ");
		buf.append(fields[8]);
		buf.append('\n');
		buf.append("number of arbitrary lookahead decisions ");
		buf.append(fields[9]);
		buf.append('\n');
		buf.append("min lookahead used in an arbitrary lookahead decision ");
		buf.append(fields[10]);
		buf.append('\n');
		buf.append("max lookahead used in an arbitrary lookahead decision ");
		buf.append(fields[11]);
		buf.append('\n');
		buf.append("average lookahead depth used in arbitrary lookahead decisions ");
		buf.append(fields[12]);
		buf.append('\n');
		buf.append("standard deviation of depth used in arbitrary lookahead decisions ");
		buf.append(fields[13]);
		buf.append('\n');
		buf.append("number of evaluated syntactic predicates ");
		buf.append(fields[14]);
		buf.append('\n');
		buf.append("min lookahead used in a syntactic predicate ");
		buf.append(fields[15]);
		buf.append('\n');
		buf.append("max lookahead used in a syntactic predicate ");
		buf.append(fields[16]);
		buf.append('\n');
		buf.append("average lookahead depth used in syntactic predicates ");
		buf.append(fields[17]);
		buf.append('\n');
		buf.append("standard deviation of depth used in syntactic predicates ");
		buf.append(fields[18]);
		buf.append('\n');
		buf.append("rule memoization cache size ");
		buf.append(fields[28]);
		buf.append('\n');
		buf.append("number of rule memoization cache hits ");
		buf.append(fields[25]);
		buf.append('\n');
		buf.append("number of rule memoization cache misses ");
		buf.append(fields[26]);
		buf.append('\n');
		buf.append("number of evaluated semantic predicates ");
		buf.append(fields[19]);
		buf.append('\n');
		buf.append("number of tokens ");
		buf.append(fields[20]);
		buf.append('\n');
		buf.append("number of hidden tokens ");
		buf.append(fields[21]);
		buf.append('\n');
		buf.append("number of char ");
		buf.append(fields[22]);
		buf.append('\n');
		buf.append("number of hidden char ");
		buf.append(fields[23]);
		buf.append('\n');
		buf.append("number of syntax errors ");
		buf.append(fields[24]);
		buf.append('\n');
		return buf.toString();
	}

	protected int[] trim(int[] X, int n) {
		if ( n<X.length ) {
			int[] trimmed = new int[n];
			System.arraycopy(X,0,trimmed,0,n);
			X = trimmed;
		}
		return X;
	}

	protected int[] toArray(List a) {
		int[] x = new int[a.size()];
		for (int i = 0; i < a.size(); i++) {
			Integer I = (Integer) a.get(i);
			x[i] = I.intValue();
		}
		return x;
	}

	/** Get num hidden tokens between i..j inclusive */
	public int getNumberOfHiddenTokens(int i, int j) {
		int n = 0;
		TokenStream input = parser.getTokenStream();
		for (int ti = i; ti<input.size() && ti <= j; ti++) {
			Token t = input.get(ti);
			if ( t.getChannel()!=Token.DEFAULT_CHANNEL ) {
				n++;
			}
		}
		return n;
	}
}
