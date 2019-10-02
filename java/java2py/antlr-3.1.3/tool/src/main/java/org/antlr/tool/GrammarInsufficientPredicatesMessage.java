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

import org.antlr.stringtemplate.StringTemplate;
import org.antlr.analysis.*;
import antlr.Token;

import java.util.*;

public class GrammarInsufficientPredicatesMessage extends Message {
	public DecisionProbe probe;
    public Map<Integer, Set<Token>> altToLocations;
	public DFAState problemState;

	public GrammarInsufficientPredicatesMessage(DecisionProbe probe,
												DFAState problemState,
												Map<Integer, Set<Token>> altToLocations)
	{
		super(ErrorManager.MSG_INSUFFICIENT_PREDICATES);
		this.probe = probe;
		this.problemState = problemState;
		this.altToLocations = altToLocations;
	}

	public String toString() {
		GrammarAST decisionASTNode = probe.dfa.getDecisionASTNode();
		line = decisionASTNode.getLine();
		column = decisionASTNode.getColumn();
		String fileName = probe.dfa.nfa.grammar.getFileName();
		if ( fileName!=null ) {
			file = fileName;
		}
		StringTemplate st = getMessageTemplate();
		// convert to string key to avoid 3.1 ST bug
		Map<String, Set<Token>> altToLocationsWithStringKey = new LinkedHashMap<String, Set<Token>>();
		List<Integer> alts = new ArrayList<Integer>();
		alts.addAll(altToLocations.keySet());
		Collections.sort(alts);
		for (Integer altI : alts) {
			altToLocationsWithStringKey.put(altI.toString(), altToLocations.get(altI));
			/*
			List<String> tokens = new ArrayList<String>();
			for (Token t : altToLocations.get(altI)) {
				tokens.add(t.toString());
			}
			Collections.sort(tokens);
			System.out.println("tokens=\n"+tokens);
			*/
		}
		st.setAttribute("altToLocations", altToLocationsWithStringKey);

		List<Label> sampleInputLabels = problemState.dfa.probe.getSampleNonDeterministicInputSequence(problemState);
		String input = problemState.dfa.probe.getInputSequenceDisplay(sampleInputLabels);
		st.setAttribute("upon", input);

		st.setAttribute("hasPredicateBlockedByAction", problemState.dfa.hasPredicateBlockedByAction);

		return super.toString(st);
	}

}
