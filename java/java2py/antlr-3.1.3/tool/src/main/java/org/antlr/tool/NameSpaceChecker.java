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

import antlr.Token;
import org.antlr.analysis.Label;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class NameSpaceChecker {
	protected Grammar grammar;

	public NameSpaceChecker(Grammar grammar) {
		this.grammar = grammar;
	}

	public void checkConflicts() {
		for (int i = CompositeGrammar.MIN_RULE_INDEX; i < grammar.composite.ruleIndexToRuleList.size(); i++) {
			Rule r = grammar.composite.ruleIndexToRuleList.elementAt(i);
			if ( r==null ) {
				continue;
			}
			// walk all labels for Rule r
			if ( r.labelNameSpace!=null ) {
				Iterator it = r.labelNameSpace.values().iterator();
				while ( it.hasNext() ) {
					Grammar.LabelElementPair pair = (Grammar.LabelElementPair) it.next();
					checkForLabelConflict(r, pair.label);
				}
			}
			// walk rule scope attributes for Rule r
			if ( r.ruleScope!=null ) {
				List attributes = r.ruleScope.getAttributes();
				for (int j = 0; j < attributes.size(); j++) {
					Attribute attribute = (Attribute) attributes.get(j);
					checkForRuleScopeAttributeConflict(r, attribute);
				}
			}
			checkForRuleDefinitionProblems(r);
			checkForRuleArgumentAndReturnValueConflicts(r);
		}
		// check all global scopes against tokens
		Iterator it = grammar.getGlobalScopes().values().iterator();
		while (it.hasNext()) {
			AttributeScope scope = (AttributeScope) it.next();
			checkForGlobalScopeTokenConflict(scope);
		}
		// check for missing rule, tokens
		lookForReferencesToUndefinedSymbols();
	}

	protected void checkForRuleArgumentAndReturnValueConflicts(Rule r) {
		if ( r.returnScope!=null ) {
			Set conflictingKeys = r.returnScope.intersection(r.parameterScope);
			if (conflictingKeys!=null) {
				for (Iterator it = conflictingKeys.iterator(); it.hasNext();) {
					String key = (String) it.next();
					ErrorManager.grammarError(
						ErrorManager.MSG_ARG_RETVAL_CONFLICT,
						grammar,
						r.tree.getToken(),
						key,
						r.name);
				}
			}
		}
	}

	protected void checkForRuleDefinitionProblems(Rule r) {
		String ruleName = r.name;
		antlr.Token ruleToken = r.tree.getToken();
		int msgID = 0;
		if ( (grammar.type==Grammar.PARSER||grammar.type==Grammar.TREE_PARSER) &&
			 Character.isUpperCase(ruleName.charAt(0)) )
		{
			msgID = ErrorManager.MSG_LEXER_RULES_NOT_ALLOWED;
        }
        else if ( grammar.type==Grammar.LEXER &&
			      Character.isLowerCase(ruleName.charAt(0)) &&
			      !r.isSynPred )
		{
			msgID = ErrorManager.MSG_PARSER_RULES_NOT_ALLOWED;
        }
		else if ( grammar.getGlobalScope(ruleName)!=null ) {
			msgID = ErrorManager.MSG_SYMBOL_CONFLICTS_WITH_GLOBAL_SCOPE;
		}
		if ( msgID!=0 ) {
			ErrorManager.grammarError(msgID, grammar, ruleToken, ruleName);
		}
	}

	/** If ref to undefined rule, give error at first occurrence.
	 * 
	 *  Give error if you cannot find the scope override on a rule reference.
	 *
	 *  If you ref ID in a combined grammar and don't define ID as a lexer rule
	 *  it is an error.
	 */
	protected void lookForReferencesToUndefinedSymbols() {
		// for each rule ref, ask if there is a rule definition
		for (Iterator iter = grammar.ruleRefs.iterator(); iter.hasNext();) {
			GrammarAST refAST = (GrammarAST)iter.next();
			Token tok = refAST.token;
			String ruleName = tok.getText();
			Rule localRule = grammar.getLocallyDefinedRule(ruleName);
			Rule rule = grammar.getRule(ruleName);
			if ( localRule==null && rule!=null ) { // imported rule?
				grammar.delegatedRuleReferences.add(rule);
				rule.imported = true;
			}
			if ( rule==null && grammar.getTokenType(ruleName)!=Label.EOF ) {
				ErrorManager.grammarError(ErrorManager.MSG_UNDEFINED_RULE_REF,
										  grammar,
										  tok,
										  ruleName);
			}
        }
		if ( grammar.type==Grammar.COMBINED ) {
			// if we're a combined grammar, we know which token IDs have no
			// associated lexer rule.
			for (Iterator iter = grammar.tokenIDRefs.iterator(); iter.hasNext();) {
				Token tok = (Token) iter.next();
				String tokenID = tok.getText();
				if ( !grammar.composite.lexerRules.contains(tokenID) &&
					 grammar.getTokenType(tokenID)!=Label.EOF )
				{
					ErrorManager.grammarWarning(ErrorManager.MSG_NO_TOKEN_DEFINITION,
												grammar,
												tok,
												tokenID);
				}
			}
		}
		// check scopes and scoped rule refs
		for (Iterator it = grammar.scopedRuleRefs.iterator(); it.hasNext();) {
			GrammarAST scopeAST = (GrammarAST)it.next(); // ^(DOT ID atom)
			Grammar scopeG = grammar.composite.getGrammar(scopeAST.getText());
			GrammarAST refAST = scopeAST.getChild(1);
			String ruleName = refAST.getText();
			if ( scopeG==null ) {
				ErrorManager.grammarError(ErrorManager.MSG_NO_SUCH_GRAMMAR_SCOPE,
										  grammar,
										  scopeAST.getToken(),
										  scopeAST.getText(),
										  ruleName);
			}
			else {
				Rule rule = grammar.getRule(scopeG.name, ruleName);
				if ( rule==null ) {
					ErrorManager.grammarError(ErrorManager.MSG_NO_SUCH_RULE_IN_SCOPE,
											  grammar,
											  scopeAST.getToken(),
											  scopeAST.getText(),
											  ruleName);
				}
			}
		}
	}

	protected void checkForGlobalScopeTokenConflict(AttributeScope scope) {
		if ( grammar.getTokenType(scope.getName())!=Label.INVALID ) {
			ErrorManager.grammarError(ErrorManager.MSG_SYMBOL_CONFLICTS_WITH_GLOBAL_SCOPE,
									  grammar, null, scope.getName());
		}
	}

	/** Check for collision of a rule-scope dynamic attribute with:
	 *  arg, return value, rule name itself.  Labels are checked elsewhere.
	 */
	public void checkForRuleScopeAttributeConflict(Rule r, Attribute attribute) {
		int msgID = 0;
		Object arg2 = null;
		String attrName = attribute.name;
		if ( r.name.equals(attrName) ) {
			msgID = ErrorManager.MSG_ATTRIBUTE_CONFLICTS_WITH_RULE;
			arg2 = r.name;
		}
		else if ( (r.returnScope!=null&&r.returnScope.getAttribute(attrName)!=null) ||
				  (r.parameterScope!=null&&r.parameterScope.getAttribute(attrName)!=null) )
		{
			msgID = ErrorManager.MSG_ATTRIBUTE_CONFLICTS_WITH_RULE_ARG_RETVAL;
			arg2 = r.name;
		}
		if ( msgID!=0 ) {
			ErrorManager.grammarError(msgID,grammar,r.tree.getToken(),attrName,arg2);
		}
	}

	/** Make sure a label doesn't conflict with another symbol.
	 *  Labels must not conflict with: rules, tokens, scope names,
	 *  return values, parameters, and rule-scope dynamic attributes
	 *  defined in surrounding rule.
	 */
	protected void checkForLabelConflict(Rule r, antlr.Token label) {
		int msgID = 0;
		Object arg2 = null;
		if ( grammar.getGlobalScope(label.getText())!=null ) {
			msgID = ErrorManager.MSG_SYMBOL_CONFLICTS_WITH_GLOBAL_SCOPE;
		}
		else if ( grammar.getRule(label.getText())!=null ) {
			msgID = ErrorManager.MSG_LABEL_CONFLICTS_WITH_RULE;
		}
		else if ( grammar.getTokenType(label.getText())!=Label.INVALID ) {
			msgID = ErrorManager.MSG_LABEL_CONFLICTS_WITH_TOKEN;
		}
		else if ( r.ruleScope!=null && r.ruleScope.getAttribute(label.getText())!=null ) {
			msgID = ErrorManager.MSG_LABEL_CONFLICTS_WITH_RULE_SCOPE_ATTRIBUTE;
			arg2 = r.name;
		}
		else if ( (r.returnScope!=null&&r.returnScope.getAttribute(label.getText())!=null) ||
				  (r.parameterScope!=null&&r.parameterScope.getAttribute(label.getText())!=null) )
		{
			msgID = ErrorManager.MSG_LABEL_CONFLICTS_WITH_RULE_ARG_RETVAL;
			arg2 = r.name;
		}
		if ( msgID!=0 ) {
			ErrorManager.grammarError(msgID,grammar,label,label.getText(),arg2);
		}
	}

	/** If type of previous label differs from new label's type, that's an error.
	 */
	public boolean checkForLabelTypeMismatch(Rule r, antlr.Token label, int type) {
		Grammar.LabelElementPair prevLabelPair =
			(Grammar.LabelElementPair)r.labelNameSpace.get(label.getText());
		if ( prevLabelPair!=null ) {
			// label already defined; if same type, no problem
			if ( prevLabelPair.type != type ) {
				String typeMismatchExpr =
					Grammar.LabelTypeToString[type]+"!="+
					Grammar.LabelTypeToString[prevLabelPair.type];
				ErrorManager.grammarError(
					ErrorManager.MSG_LABEL_TYPE_CONFLICT,
					grammar,
					label,
					label.getText(),
					typeMismatchExpr);
				return true;
			}
		}
		return false;
	}
}
