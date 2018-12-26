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
import org.antlr.analysis.DecisionProbe;
import org.antlr.analysis.DFAState;
import org.antlr.analysis.NFAState;
import org.antlr.analysis.SemanticContext;
import antlr.Token;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

/** Reports a potential parsing issue with a decision; the decision is
 *  nondeterministic in some way.
 */
public class GrammarUnreachableAltsMessage extends Message {
	public DecisionProbe probe;
    public List alts;

	public GrammarUnreachableAltsMessage(DecisionProbe probe,
										 List alts)
	{
		super(ErrorManager.MSG_UNREACHABLE_ALTS);
		this.probe = probe;
		this.alts = alts;
		// flip msg ID if alts are actually token refs in Tokens rule
		if ( probe.dfa.isTokensRuleDecision() ) {
			setMessageID(ErrorManager.MSG_UNREACHABLE_TOKENS);
		}
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

		if ( probe.dfa.isTokensRuleDecision() ) {
			// alts are token rules, convert to the names instead of numbers
			for (int i = 0; i < alts.size(); i++) {
				Integer altI = (Integer) alts.get(i);
				String tokenName =
					probe.getTokenNameForTokensRuleAlt(altI.intValue());
				// reset the line/col to the token definition
				NFAState ruleStart =
					probe.dfa.nfa.grammar.getRuleStartState(tokenName);
				line = ruleStart.associatedASTNode.getLine();
				column = ruleStart.associatedASTNode.getColumn();
				st.setAttribute("tokens", tokenName);
			}
		}
		else {
			// regular alt numbers, show the alts
			st.setAttribute("alts", alts);
		}

		return super.toString(st);
	}

}
