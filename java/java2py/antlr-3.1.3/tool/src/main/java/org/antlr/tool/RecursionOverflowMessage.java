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

/** Indicates recursion overflow.  A DFA state tried add an NFA configuration
 *  with NFA state p that was mentioned in its stack context too many times.
 */
public class RecursionOverflowMessage extends Message {
	public DecisionProbe probe;
	public DFAState sampleBadState;
	public int alt;
	public Collection targetRules;
	public Collection callSiteStates;

	public RecursionOverflowMessage(DecisionProbe probe,
									DFAState sampleBadState,
									int alt,
									Collection targetRules,
									Collection callSiteStates)
	{
		super(ErrorManager.MSG_RECURSION_OVERLOW);
		this.probe = probe;
		this.sampleBadState = sampleBadState;
		this.alt = alt;
		this.targetRules = targetRules;
		this.callSiteStates = callSiteStates;
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
		st.setAttribute("targetRules", targetRules);
		st.setAttribute("alt", alt);
		st.setAttribute("callSiteStates", callSiteStates);

		List labels =
			probe.getSampleNonDeterministicInputSequence(sampleBadState);
		String input = probe.getInputSequenceDisplay(labels);
		st.setAttribute("input", input);

		return super.toString(st);
	}

}
