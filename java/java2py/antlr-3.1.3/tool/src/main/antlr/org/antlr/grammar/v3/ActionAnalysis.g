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

/** We need to set Rule.referencedPredefinedRuleAttributes before
 *  code generation.  This filter looks at an action in context of
 *  its rule and outer alternative number and figures out which
 *  rules have predefined prefs referenced.  I need this so I can
 *  remove unusued labels.  This also tracks, for labeled rules,
 *  which are referenced by actions.
 */
lexer grammar ActionAnalysis;
options {
  filter=true;  // try all non-fragment rules in order specified
}

@header {
package org.antlr.grammar.v3;
import org.antlr.runtime.*;
import org.antlr.tool.*;
}

@members {
Rule enclosingRule;
Grammar grammar;
antlr.Token actionToken;
int outerAltNum = 0;

	public ActionAnalysis(Grammar grammar, String ruleName, GrammarAST actionAST)
	{
		this(new ANTLRStringStream(actionAST.token.getText()));
		this.grammar = grammar;
	    this.enclosingRule = grammar.getLocallyDefinedRule(ruleName);
	    this.actionToken = actionAST.token;
	    this.outerAltNum = actionAST.outerAltNum;
	}

public void analyze() {
	// System.out.println("###\naction="+actionToken);
	Token t;
	do {
		t = nextToken();
	} while ( t.getType()!= Token.EOF );
}
}

/**	$x.y	x is enclosing rule or rule ref or rule label
 *			y is a return value, parameter, or predefined property.
 */
X_Y :	'$' x=ID '.' y=ID {enclosingRule!=null}?
		{
		AttributeScope scope = null;
		String refdRuleName = null;
		if ( $x.text.equals(enclosingRule.name) ) {
			// ref to enclosing rule.
			refdRuleName = $x.text;
			scope = enclosingRule.getLocalAttributeScope($y.text);
		}
		else if ( enclosingRule.getRuleLabel($x.text)!=null ) {
			// ref to rule label
			Grammar.LabelElementPair pair = enclosingRule.getRuleLabel($x.text);
			pair.actionReferencesLabel = true;
			refdRuleName = pair.referencedRuleName;
			Rule refdRule = grammar.getRule(refdRuleName);
			if ( refdRule!=null ) {
				scope = refdRule.getLocalAttributeScope($y.text);
			}
		}
		else if ( enclosingRule.getRuleRefsInAlt(x.getText(), outerAltNum)!=null ) {
			// ref to rule referenced in this alt
			refdRuleName = $x.text;
			Rule refdRule = grammar.getRule(refdRuleName);
			if ( refdRule!=null ) {
				scope = refdRule.getLocalAttributeScope($y.text);
			}
		}
		if ( scope!=null &&
			 (scope.isPredefinedRuleScope||scope.isPredefinedLexerRuleScope) )
		{
			grammar.referenceRuleLabelPredefinedAttribute(refdRuleName);
			//System.out.println("referenceRuleLabelPredefinedAttribute for "+refdRuleName);
		}
		}
	;

/** $x	x is an isolated rule label.  Just record that the label was referenced */
X	:	'$' x=ID {enclosingRule!=null && enclosingRule.getRuleLabel($x.text)!=null}?
		{
			Grammar.LabelElementPair pair = enclosingRule.getRuleLabel($x.text);
			pair.actionReferencesLabel = true;
		}
	;
	
/** $y	y is a return value, parameter, or predefined property of current rule */
Y	:	'$' ID {enclosingRule!=null && enclosingRule.getLocalAttributeScope($ID.text)!=null}?
		{
			AttributeScope scope = enclosingRule.getLocalAttributeScope($ID.text);
			if ( scope!=null &&
				 (scope.isPredefinedRuleScope||scope.isPredefinedLexerRuleScope) )
			{
				grammar.referenceRuleLabelPredefinedAttribute(enclosingRule.name);
				//System.out.println("referenceRuleLabelPredefinedAttribute for "+$ID.text);
			}
		}
	;
	
fragment
ID  :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'_'|'0'..'9')*
    ;
