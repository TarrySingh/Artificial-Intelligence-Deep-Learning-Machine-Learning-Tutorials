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
package org.antlr.tool;

import org.antlr.analysis.DFA;
import org.antlr.runtime.misc.Stats;
import org.antlr.misc.Utils;

import java.util.*;

public class GrammarReport {
	/** Because I may change the stats, I need to track that for later
	 *  computations to be consistent.
	 */
	public static final String Version = "4";
	public static final String GRAMMAR_STATS_FILENAME = "grammar.stats";
	public static final int NUM_GRAMMAR_STATS = 41;

	public static final String newline = System.getProperty("line.separator");

	public Grammar grammar;

	public GrammarReport(Grammar grammar) {
		this.grammar = grammar;
	}

	/** Create a single-line stats report about this grammar suitable to
	 *  send to the notify page at antlr.org
	 */
	public String toNotifyString() {
		StringBuffer buf = new StringBuffer();
		buf.append(Version);
		buf.append('\t');
		buf.append(grammar.name);
		buf.append('\t');
		buf.append(grammar.getGrammarTypeString());
		buf.append('\t');
		buf.append(grammar.getOption("language"));
		int totalNonSynPredProductions = 0;
		int totalNonSynPredRules = 0;
		Collection rules = grammar.getRules();
		for (Iterator it = rules.iterator(); it.hasNext();) {
			Rule r = (Rule) it.next();
			if ( !r.name.toUpperCase()
				.startsWith(Grammar.SYNPRED_RULE_PREFIX.toUpperCase()) )
			{
				totalNonSynPredProductions += r.numberOfAlts;
				totalNonSynPredRules++;
			}
		}
		buf.append('\t');
		buf.append(totalNonSynPredRules);
		buf.append('\t');
		buf.append(totalNonSynPredProductions);
		int numACyclicDecisions =
			grammar.getNumberOfDecisions()-grammar.getNumberOfCyclicDecisions();
		int[] depths = new int[numACyclicDecisions];
		int[] acyclicDFAStates = new int[numACyclicDecisions];
		int[] cyclicDFAStates = new int[grammar.getNumberOfCyclicDecisions()];
		int acyclicIndex = 0;
		int cyclicIndex = 0;
		int numLL1 = 0;
		int numDec = 0;
		for (int i=1; i<=grammar.getNumberOfDecisions(); i++) {
			Grammar.Decision d = grammar.getDecision(i);
			if( d.dfa==null ) {
				continue;
			}
			numDec++;
			if ( !d.dfa.isCyclic() ) {
				int maxk = d.dfa.getMaxLookaheadDepth();
				if ( maxk==1 ) {
					numLL1++;
				}
				depths[acyclicIndex] = maxk;
				acyclicDFAStates[acyclicIndex] = d.dfa.getNumberOfStates();
				acyclicIndex++;
			}
			else {
				cyclicDFAStates[cyclicIndex] = d.dfa.getNumberOfStates();
				cyclicIndex++;
			}
		}
		buf.append('\t');
		buf.append(numDec);
		buf.append('\t');
		buf.append(grammar.getNumberOfCyclicDecisions());
		buf.append('\t');
		buf.append(numLL1);
		buf.append('\t');
		buf.append(Stats.min(depths));
		buf.append('\t');
		buf.append(Stats.max(depths));
		buf.append('\t');
		buf.append(Stats.avg(depths));
		buf.append('\t');
		buf.append(Stats.stddev(depths));
		buf.append('\t');
		buf.append(Stats.min(acyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.max(acyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.avg(acyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.stddev(acyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.sum(acyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.min(cyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.max(cyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.avg(cyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.stddev(cyclicDFAStates));
		buf.append('\t');
		buf.append(Stats.sum(cyclicDFAStates));
		buf.append('\t');
		buf.append(grammar.getTokenTypes().size());
		buf.append('\t');
		buf.append(grammar.DFACreationWallClockTimeInMS);
		buf.append('\t');
		buf.append(grammar.numberOfSemanticPredicates);
		buf.append('\t');
		buf.append(grammar.numberOfManualLookaheadOptions);
		buf.append('\t');
		buf.append(grammar.setOfNondeterministicDecisionNumbers.size());
		buf.append('\t');
		buf.append(grammar.setOfNondeterministicDecisionNumbersResolvedWithPredicates.size());
		buf.append('\t');
		buf.append(grammar.setOfDFAWhoseAnalysisTimedOut.size());
		buf.append('\t');
		buf.append(ErrorManager.getErrorState().errors);
		buf.append('\t');
		buf.append(ErrorManager.getErrorState().warnings);
		buf.append('\t');
		buf.append(ErrorManager.getErrorState().infos);
		buf.append('\t');
		Map synpreds = grammar.getSyntacticPredicates();
		int num_synpreds = synpreds!=null ? synpreds.size() : 0;
		buf.append(num_synpreds);
		buf.append('\t');
		buf.append(grammar.blocksWithSynPreds.size());
		buf.append('\t');
		buf.append(grammar.decisionsWhoseDFAsUsesSynPreds.size());
		buf.append('\t');
		buf.append(grammar.blocksWithSemPreds.size());
		buf.append('\t');
		buf.append(grammar.decisionsWhoseDFAsUsesSemPreds.size());
		buf.append('\t');
		String output = (String)grammar.getOption("output");
		if ( output==null ) {
			output = "none";
		}
		buf.append(output);
		buf.append('\t');
		Object k = grammar.getOption("k");
		if ( k==null ) {
			k = "none";
		}
		buf.append(k);
		buf.append('\t');
		String backtrack = (String)grammar.getOption("backtrack");
		if ( backtrack==null ) {
			backtrack = "false";
		}
		buf.append(backtrack);
		return buf.toString();
	}

	public String getBacktrackingReport() {
		StringBuffer buf = new StringBuffer();
		buf.append("Backtracking report:");
		buf.append(newline);
		buf.append("Number of decisions that backtrack: ");
		buf.append(grammar.decisionsWhoseDFAsUsesSynPreds.size());
		buf.append(newline);
		buf.append(getDFALocations(grammar.decisionsWhoseDFAsUsesSynPreds));
		return buf.toString();
	}

	public String getAnalysisTimeoutReport() {
		StringBuffer buf = new StringBuffer();
		buf.append("NFA conversion early termination report:");
		buf.append(newline);
		buf.append("Number of NFA conversions that terminated early: ");
		buf.append(grammar.setOfDFAWhoseAnalysisTimedOut.size());
		buf.append(newline);
		buf.append(getDFALocations(grammar.setOfDFAWhoseAnalysisTimedOut));
		return buf.toString();
	}

	protected String getDFALocations(Set dfas) {
		Set decisions = new HashSet();
		StringBuffer buf = new StringBuffer();
		Iterator it = dfas.iterator();
		while ( it.hasNext() ) {
			DFA dfa = (DFA) it.next();
			// if we aborted a DFA and redid with k=1, the backtrackin
			if ( decisions.contains(Utils.integer(dfa.decisionNumber)) ) {
				continue;
			}
			decisions.add(Utils.integer(dfa.decisionNumber));
			buf.append("Rule ");
			buf.append(dfa.decisionNFAStartState.enclosingRule.name);
			buf.append(" decision ");
			buf.append(dfa.decisionNumber);
			buf.append(" location ");
			GrammarAST decisionAST =
				dfa.decisionNFAStartState.associatedASTNode;
			buf.append(decisionAST.getLine());
			buf.append(":");
			buf.append(decisionAST.getColumn());
			buf.append(newline);
		}
		return buf.toString();
	}

	/** Given a stats line suitable for sending to the antlr.org site,
	 *  return a human-readable version.  Return null if there is a
	 *  problem with the data.
	 */
	public String toString() {
		return toString(toNotifyString());
	}

	protected static String[] decodeReportData(String data) {
		String[] fields = new String[NUM_GRAMMAR_STATS];
		StringTokenizer st = new StringTokenizer(data, "\t");
		int i = 0;
		while ( st.hasMoreTokens() ) {
			fields[i] = st.nextToken();
			i++;
		}
		if ( i!=NUM_GRAMMAR_STATS ) {
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
		buf.append("ANTLR Grammar Report; Stats Version ");
		buf.append(fields[0]);
		buf.append('\n');
		buf.append("Grammar: ");
		buf.append(fields[1]);
		buf.append('\n');
		buf.append("Type: ");
		buf.append(fields[2]);
		buf.append('\n');
		buf.append("Target language: ");
		buf.append(fields[3]);
		buf.append('\n');
		buf.append("Output: ");
		buf.append(fields[38]);
		buf.append('\n');
		buf.append("Grammar option k: ");
		buf.append(fields[39]);
		buf.append('\n');
		buf.append("Grammar option backtrack: ");
		buf.append(fields[40]);
		buf.append('\n');
		buf.append("Rules: ");
		buf.append(fields[4]);
		buf.append('\n');
		buf.append("Productions: ");
		buf.append(fields[5]);
		buf.append('\n');
		buf.append("Decisions: ");
		buf.append(fields[6]);
		buf.append('\n');
		buf.append("Cyclic DFA decisions: ");
		buf.append(fields[7]);
		buf.append('\n');
		buf.append("LL(1) decisions: "); buf.append(fields[8]);
		buf.append('\n');
		buf.append("Min fixed k: "); buf.append(fields[9]);
		buf.append('\n');
		buf.append("Max fixed k: "); buf.append(fields[10]);
		buf.append('\n');
		buf.append("Average fixed k: "); buf.append(fields[11]);
		buf.append('\n');
		buf.append("Standard deviation of fixed k: "); buf.append(fields[12]);
		buf.append('\n');
		buf.append("Min acyclic DFA states: "); buf.append(fields[13]);
		buf.append('\n');
		buf.append("Max acyclic DFA states: "); buf.append(fields[14]);
		buf.append('\n');
		buf.append("Average acyclic DFA states: "); buf.append(fields[15]);
		buf.append('\n');
		buf.append("Standard deviation of acyclic DFA states: "); buf.append(fields[16]);
		buf.append('\n');
		buf.append("Total acyclic DFA states: "); buf.append(fields[17]);
		buf.append('\n');
		buf.append("Min cyclic DFA states: "); buf.append(fields[18]);
		buf.append('\n');
		buf.append("Max cyclic DFA states: "); buf.append(fields[19]);
		buf.append('\n');
		buf.append("Average cyclic DFA states: "); buf.append(fields[20]);
		buf.append('\n');
		buf.append("Standard deviation of cyclic DFA states: "); buf.append(fields[21]);
		buf.append('\n');
		buf.append("Total cyclic DFA states: "); buf.append(fields[22]);
		buf.append('\n');
		buf.append("Vocabulary size: ");
		buf.append(fields[23]);
		buf.append('\n');
		buf.append("DFA creation time in ms: ");
		buf.append(fields[24]);
		buf.append('\n');
		buf.append("Number of semantic predicates found: ");
		buf.append(fields[25]);
		buf.append('\n');
		buf.append("Number of manual fixed lookahead k=value options: ");
		buf.append(fields[26]);
		buf.append('\n');
		buf.append("Number of nondeterministic decisions: ");
		buf.append(fields[27]);
		buf.append('\n');
		buf.append("Number of nondeterministic decisions resolved with predicates: ");
		buf.append(fields[28]);
		buf.append('\n');
		buf.append("Number of DFA conversions terminated early: ");
		buf.append(fields[29]);
		buf.append('\n');
		buf.append("Number of errors: ");
		buf.append(fields[30]);
		buf.append('\n');
		buf.append("Number of warnings: ");
		buf.append(fields[31]);
		buf.append('\n');
		buf.append("Number of infos: ");
		buf.append(fields[32]);
		buf.append('\n');
		buf.append("Number of syntactic predicates found: ");
		buf.append(fields[33]);
		buf.append('\n');
		buf.append("Decisions with syntactic predicates: ");
		buf.append(fields[34]);
		buf.append('\n');
		buf.append("Decision DFAs using syntactic predicates: ");
		buf.append(fields[35]);
		buf.append('\n');
		buf.append("Decisions with semantic predicates: ");
		buf.append(fields[36]);
		buf.append('\n');
		buf.append("Decision DFAs using semantic predicates: ");
		buf.append(fields[37]);
		buf.append('\n');
		return buf.toString();
	}

}
