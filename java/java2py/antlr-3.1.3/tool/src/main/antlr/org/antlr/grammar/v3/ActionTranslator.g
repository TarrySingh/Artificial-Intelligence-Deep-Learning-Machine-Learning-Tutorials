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

lexer grammar ActionTranslator;
options {
  filter=true;  // try all non-fragment rules in order specified
  // output=template;  TODO: can we make tokens return templates somehow?
}

@header {
package org.antlr.grammar.v3;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.runtime.*;
import org.antlr.tool.*;
import org.antlr.codegen.*;

import org.antlr.runtime.*;
import java.util.List;
import java.util.ArrayList;
import org.antlr.grammar.v2.ANTLRParser;

}

@members {
public List chunks = new ArrayList();
Rule enclosingRule;
int outerAltNum;
Grammar grammar;
CodeGenerator generator;
antlr.Token actionToken;

	public ActionTranslator(CodeGenerator generator,
								 String ruleName,
								 GrammarAST actionAST)
	{
		this(new ANTLRStringStream(actionAST.token.getText()));
		this.generator = generator;
		this.grammar = generator.grammar;
	    this.enclosingRule = grammar.getLocallyDefinedRule(ruleName);
	    this.actionToken = actionAST.token;
	    this.outerAltNum = actionAST.outerAltNum;
	}

	public ActionTranslator(CodeGenerator generator,
								 String ruleName,
								 antlr.Token actionToken,
								 int outerAltNum)
	{
		this(new ANTLRStringStream(actionToken.getText()));
		this.generator = generator;
		grammar = generator.grammar;
	    this.enclosingRule = grammar.getRule(ruleName);
	    this.actionToken = actionToken;
		this.outerAltNum = outerAltNum;
	}

/** Return a list of strings and StringTemplate objects that
 *  represent the translated action.
 */
public List translateToChunks() {
	// System.out.println("###\naction="+action);
	Token t;
	do {
		t = nextToken();
	} while ( t.getType()!= Token.EOF );
	return chunks;
}

public String translate() {
	List theChunks = translateToChunks();
	//System.out.println("chunks="+a.chunks);
	StringBuffer buf = new StringBuffer();
	for (int i = 0; i < theChunks.size(); i++) {
		Object o = (Object) theChunks.get(i);
		buf.append(o);
	}
	//System.out.println("translated: "+buf.toString());
	return buf.toString();
}

public List translateAction(String action) {
	String rname = null;
	if ( enclosingRule!=null ) {
		rname = enclosingRule.name;
	}
	ActionTranslator translator =
		new ActionTranslator(generator,
								  rname,
								  new antlr.CommonToken(ANTLRParser.ACTION,action),outerAltNum);
    return translator.translateToChunks();
}

public boolean isTokenRefInAlt(String id) {
    return enclosingRule.getTokenRefsInAlt(id, outerAltNum)!=null;
}
public boolean isRuleRefInAlt(String id) {
    return enclosingRule.getRuleRefsInAlt(id, outerAltNum)!=null;
}
public Grammar.LabelElementPair getElementLabel(String id) {
    return enclosingRule.getLabel(id);
}

public void checkElementRefUniqueness(String ref, boolean isToken) {
		List refs = null;
		if ( isToken ) {
		    refs = enclosingRule.getTokenRefsInAlt(ref, outerAltNum);
		}
		else {
		    refs = enclosingRule.getRuleRefsInAlt(ref, outerAltNum);
		}
		if ( refs!=null && refs.size()>1 ) {
			ErrorManager.grammarError(ErrorManager.MSG_NONUNIQUE_REF,
									  grammar,
									  actionToken,
									  ref);
		}
}

/** For \$rulelabel.name, return the Attribute found for name.  It
 *  will be a predefined property or a return value.
 */
public Attribute getRuleLabelAttribute(String ruleName, String attrName) {
	Rule r = grammar.getRule(ruleName);
	AttributeScope scope = r.getLocalAttributeScope(attrName);
	if ( scope!=null && !scope.isParameterScope ) {
		return scope.getAttribute(attrName);
	}
	return null;
}

AttributeScope resolveDynamicScope(String scopeName) {
	if ( grammar.getGlobalScope(scopeName)!=null ) {
		return grammar.getGlobalScope(scopeName);
	}
	Rule scopeRule = grammar.getRule(scopeName);
	if ( scopeRule!=null ) {
		return scopeRule.ruleScope;
	}
	return null; // not a valid dynamic scope
}

protected StringTemplate template(String name) {
	StringTemplate st = generator.getTemplates().getInstanceOf(name);
	chunks.add(st);
	return st;
}


}

/**	$x.y	x is enclosing rule, y is a return value, parameter, or
 * 			predefined property.
 *
 * 			r[int i] returns [int j]
 * 				:	{$r.i, $r.j, $r.start, $r.stop, $r.st, $r.tree}
 * 				;
 */
SET_ENCLOSING_RULE_SCOPE_ATTR
	:	'$' x=ID '.' y=ID WS? '=' expr=ATTR_VALUE_EXPR ';'
							{enclosingRule!=null &&
	                         $x.text.equals(enclosingRule.name) &&
	                         enclosingRule.getLocalAttributeScope($y.text)!=null}?
		//{System.out.println("found \$rule.attr");}
		{
		StringTemplate st = null;
		AttributeScope scope = enclosingRule.getLocalAttributeScope($y.text);
		if ( scope.isPredefinedRuleScope ) {
			if ( $y.text.equals("st") || $y.text.equals("tree") ) {
				st = template("ruleSetPropertyRef_"+$y.text);
				grammar.referenceRuleLabelPredefinedAttribute($x.text);
				st.setAttribute("scope", $x.text);
				st.setAttribute("attr", $y.text);
				st.setAttribute("expr", translateAction($expr.text));
			} else {
				ErrorManager.grammarError(ErrorManager.MSG_WRITE_TO_READONLY_ATTR,
										  grammar,
										  actionToken,
										  $x.text,
										  $y.text);
			}
		}
	    else if ( scope.isPredefinedLexerRuleScope ) {
	    	// this is a better message to emit than the previous one...
			ErrorManager.grammarError(ErrorManager.MSG_WRITE_TO_READONLY_ATTR,
									  grammar,
									  actionToken,
									  $x.text,
									  $y.text);
	    }
		else if ( scope.isParameterScope ) {
			st = template("parameterSetAttributeRef");
			st.setAttribute("attr", scope.getAttribute($y.text));
			st.setAttribute("expr", translateAction($expr.text));
		}
		else { // must be return value
			st = template("returnSetAttributeRef");
			st.setAttribute("ruleDescriptor", enclosingRule);
			st.setAttribute("attr", scope.getAttribute($y.text));
			st.setAttribute("expr", translateAction($expr.text));
		}
		}
	;
ENCLOSING_RULE_SCOPE_ATTR
	:	'$' x=ID '.' y=ID	{enclosingRule!=null &&
	                         $x.text.equals(enclosingRule.name) &&
	                         enclosingRule.getLocalAttributeScope($y.text)!=null}?
		//{System.out.println("found \$rule.attr");}
		{
		if ( isRuleRefInAlt($x.text)  ) {
			ErrorManager.grammarError(ErrorManager.MSG_RULE_REF_AMBIG_WITH_RULE_IN_ALT,
									  grammar,
									  actionToken,
									  $x.text);
		}
		StringTemplate st = null;
		AttributeScope scope = enclosingRule.getLocalAttributeScope($y.text);
		if ( scope.isPredefinedRuleScope ) {
			st = template("rulePropertyRef_"+$y.text);
			grammar.referenceRuleLabelPredefinedAttribute($x.text);
			st.setAttribute("scope", $x.text);
			st.setAttribute("attr", $y.text);
		}
	    else if ( scope.isPredefinedLexerRuleScope ) {
	    	// perhaps not the most precise error message to use, but...
			ErrorManager.grammarError(ErrorManager.MSG_RULE_HAS_NO_ARGS,
									  grammar,
									  actionToken,
									  $x.text);
	    }
		else if ( scope.isParameterScope ) {
			st = template("parameterAttributeRef");
			st.setAttribute("attr", scope.getAttribute($y.text));
		}
		else { // must be return value
			st = template("returnAttributeRef");
			st.setAttribute("ruleDescriptor", enclosingRule);
			st.setAttribute("attr", scope.getAttribute($y.text));
		}
		}
	;

/** Setting $tokenlabel.attr or $tokenref.attr where attr is predefined property of a token is an error. */
SET_TOKEN_SCOPE_ATTR
	:	'$' x=ID '.' y=ID WS? '='
							 {enclosingRule!=null && input.LA(1)!='=' &&
	                         (enclosingRule.getTokenLabel($x.text)!=null||
	                          isTokenRefInAlt($x.text)) &&
	                         AttributeScope.tokenScope.getAttribute($y.text)!=null}?
		//{System.out.println("found \$tokenlabel.attr or \$tokenref.attr");}
		{
		ErrorManager.grammarError(ErrorManager.MSG_WRITE_TO_READONLY_ATTR,
								  grammar,
								  actionToken,
								  $x.text,
								  $y.text);
		}
	;

/** $tokenlabel.attr or $tokenref.attr where attr is predefined property of a token.
 *  If in lexer grammar, only translate for strings and tokens (rule refs)
 */
TOKEN_SCOPE_ATTR
	:	'$' x=ID '.' y=ID	{enclosingRule!=null &&
	                         (enclosingRule.getTokenLabel($x.text)!=null||
	                          isTokenRefInAlt($x.text)) &&
	                         AttributeScope.tokenScope.getAttribute($y.text)!=null &&
	                         (grammar.type!=Grammar.LEXER ||
	                         getElementLabel($x.text).elementRef.token.getType()==ANTLRParser.TOKEN_REF ||
	                         getElementLabel($x.text).elementRef.token.getType()==ANTLRParser.STRING_LITERAL)}?
		// {System.out.println("found \$tokenlabel.attr or \$tokenref.attr");}
		{
		String label = $x.text;
		if ( enclosingRule.getTokenLabel($x.text)==null ) {
			// \$tokenref.attr  gotta get old label or compute new one
			checkElementRefUniqueness($x.text, true);
			label = enclosingRule.getElementLabel($x.text, outerAltNum, generator);
			if ( label==null ) {
				ErrorManager.grammarError(ErrorManager.MSG_FORWARD_ELEMENT_REF,
										  grammar,
										  actionToken,
										  "\$"+$x.text+"."+$y.text);
				label = $x.text;
			}
		}
		StringTemplate st = template("tokenLabelPropertyRef_"+$y.text);
		st.setAttribute("scope", label);
		st.setAttribute("attr", AttributeScope.tokenScope.getAttribute($y.text));
		}
	;

/** Setting $rulelabel.attr or $ruleref.attr where attr is a predefined property is an error
 *  This must also fail, if we try to access a local attribute's field, like $tree.scope = localObject
 *  That must be handled by LOCAL_ATTR below. ANTLR only concerns itself with the top-level scope
 *  attributes declared in scope {} or parameters, return values and the like.
 */
SET_RULE_SCOPE_ATTR
@init {
Grammar.LabelElementPair pair=null;
String refdRuleName=null;
}
	:	'$' x=ID '.' y=ID WS? '=' {enclosingRule!=null && input.LA(1)!='='}?
		{
		pair = enclosingRule.getRuleLabel($x.text);
		refdRuleName = $x.text;
		if ( pair!=null ) {
			refdRuleName = pair.referencedRuleName;
		}
		}
		// supercomplicated because I can't exec the above action.
		// This asserts that if it's a label or a ref to a rule proceed but only if the attribute
		// is valid for that rule's scope
		{(enclosingRule.getRuleLabel($x.text)!=null || isRuleRefInAlt($x.text)) &&
	      getRuleLabelAttribute(enclosingRule.getRuleLabel($x.text)!=null?enclosingRule.getRuleLabel($x.text).referencedRuleName:$x.text,$y.text)!=null}?
		//{System.out.println("found set \$rulelabel.attr or \$ruleref.attr: "+$x.text+"."+$y.text);}
		{
		ErrorManager.grammarError(ErrorManager.MSG_WRITE_TO_READONLY_ATTR,
								  grammar,
								  actionToken,
								  $x.text,
								  $y.text);
		}
	;

/** $rulelabel.attr or $ruleref.attr where attr is a predefined property*/
RULE_SCOPE_ATTR
@init {
Grammar.LabelElementPair pair=null;
String refdRuleName=null;
}
	:	'$' x=ID '.' y=ID {enclosingRule!=null}?
		{
		pair = enclosingRule.getRuleLabel($x.text);
		refdRuleName = $x.text;
		if ( pair!=null ) {
			refdRuleName = pair.referencedRuleName;
		}
		}
		// supercomplicated because I can't exec the above action.
		// This asserts that if it's a label or a ref to a rule proceed but only if the attribute
		// is valid for that rule's scope
		{(enclosingRule.getRuleLabel($x.text)!=null || isRuleRefInAlt($x.text)) &&
	      getRuleLabelAttribute(enclosingRule.getRuleLabel($x.text)!=null?enclosingRule.getRuleLabel($x.text).referencedRuleName:$x.text,$y.text)!=null}?
		//{System.out.println("found \$rulelabel.attr or \$ruleref.attr: "+$x.text+"."+$y.text);}
		{
		String label = $x.text;
		if ( pair==null ) {
			// \$ruleref.attr  gotta get old label or compute new one
			checkElementRefUniqueness($x.text, false);
			label = enclosingRule.getElementLabel($x.text, outerAltNum, generator);
			if ( label==null ) {
				ErrorManager.grammarError(ErrorManager.MSG_FORWARD_ELEMENT_REF,
										  grammar,
										  actionToken,
										  "\$"+$x.text+"."+$y.text);
				label = $x.text;
			}
		}
		StringTemplate st;
		Rule refdRule = grammar.getRule(refdRuleName);
		AttributeScope scope = refdRule.getLocalAttributeScope($y.text);
		if ( scope.isPredefinedRuleScope ) {
			st = template("ruleLabelPropertyRef_"+$y.text);
			grammar.referenceRuleLabelPredefinedAttribute(refdRuleName);
			st.setAttribute("scope", label);
			st.setAttribute("attr", $y.text);
		}
		else if ( scope.isPredefinedLexerRuleScope ) {
			st = template("lexerRuleLabelPropertyRef_"+$y.text);
			grammar.referenceRuleLabelPredefinedAttribute(refdRuleName);
			st.setAttribute("scope", label);
			st.setAttribute("attr", $y.text);
		}
		else if ( scope.isParameterScope ) {
			// TODO: error!
		}
		else {
			st = template("ruleLabelRef");
			st.setAttribute("referencedRule", refdRule);
			st.setAttribute("scope", label);
			st.setAttribute("attr", scope.getAttribute($y.text));
		}
		}
	;


/** $label	either a token label or token/rule list label like label+=expr */
LABEL_REF
	:	'$' ID {enclosingRule!=null &&
	            getElementLabel($ID.text)!=null &&
		        enclosingRule.getRuleLabel($ID.text)==null}?
		// {System.out.println("found \$label");}
		{
		StringTemplate st;
		Grammar.LabelElementPair pair = getElementLabel($ID.text);
		if ( pair.type==Grammar.RULE_LIST_LABEL ||
             pair.type==Grammar.TOKEN_LIST_LABEL ||
             pair.type==Grammar.WILDCARD_TREE_LIST_LABEL )
        {
			st = template("listLabelRef");
		}
		else {
			st = template("tokenLabelRef");
		}
		st.setAttribute("label", $ID.text);
		}
	;

/** $tokenref in a non-lexer grammar */
ISOLATED_TOKEN_REF
	:	'$' ID	{grammar.type!=Grammar.LEXER && enclosingRule!=null && isTokenRefInAlt($ID.text)}?
		//{System.out.println("found \$tokenref");}
		{
		String label = enclosingRule.getElementLabel($ID.text, outerAltNum, generator);
		checkElementRefUniqueness($ID.text, true);
		if ( label==null ) {
			ErrorManager.grammarError(ErrorManager.MSG_FORWARD_ELEMENT_REF,
									  grammar,
									  actionToken,
									  $ID.text);
		}
		else {
			StringTemplate st = template("tokenLabelRef");
			st.setAttribute("label", label);
		}
		}
	;

/** $lexerruleref from within the lexer */
ISOLATED_LEXER_RULE_REF
	:	'$' ID	{grammar.type==Grammar.LEXER &&
	             enclosingRule!=null &&
	             isRuleRefInAlt($ID.text)}?
		//{System.out.println("found \$lexerruleref");}
		{
		String label = enclosingRule.getElementLabel($ID.text, outerAltNum, generator);
		checkElementRefUniqueness($ID.text, false);
		if ( label==null ) {
			ErrorManager.grammarError(ErrorManager.MSG_FORWARD_ELEMENT_REF,
									  grammar,
									  actionToken,
									  $ID.text);
		}
		else {
			StringTemplate st = template("lexerRuleLabel");
			st.setAttribute("label", label);
		}
		}
	;

/**  $y 	return value, parameter, predefined rule property, or token/rule
 *          reference within enclosing rule's outermost alt.
 *          y must be a "local" reference; i.e., it must be referring to
 *          something defined within the enclosing rule.
 *
 * 			r[int i] returns [int j]
 * 				:	{$i, $j, $start, $stop, $st, $tree}
 *              ;
 *
 *	TODO: this might get the dynamic scope's elements too.!!!!!!!!!
 */
SET_LOCAL_ATTR
	:	'$' ID WS? '=' expr=ATTR_VALUE_EXPR ';' {enclosingRule!=null
													&& enclosingRule.getLocalAttributeScope($ID.text)!=null
													&& !enclosingRule.getLocalAttributeScope($ID.text).isPredefinedLexerRuleScope}?
		//{System.out.println("found set \$localattr");}
		{
		StringTemplate st;
		AttributeScope scope = enclosingRule.getLocalAttributeScope($ID.text);
		if ( scope.isPredefinedRuleScope ) {
			if ($ID.text.equals("tree") || $ID.text.equals("st")) {
				st = template("ruleSetPropertyRef_"+$ID.text);
				grammar.referenceRuleLabelPredefinedAttribute(enclosingRule.name);
				st.setAttribute("scope", enclosingRule.name);
				st.setAttribute("attr", $ID.text);
				st.setAttribute("expr", translateAction($expr.text));
			} else {
				ErrorManager.grammarError(ErrorManager.MSG_WRITE_TO_READONLY_ATTR,
										 grammar,
										 actionToken,
										 $ID.text,
										 "");
			}
		}
		else if ( scope.isParameterScope ) {
			st = template("parameterSetAttributeRef");
			st.setAttribute("attr", scope.getAttribute($ID.text));
			st.setAttribute("expr", translateAction($expr.text));
		}
		else {
			st = template("returnSetAttributeRef");
			st.setAttribute("ruleDescriptor", enclosingRule);
			st.setAttribute("attr", scope.getAttribute($ID.text));
			st.setAttribute("expr", translateAction($expr.text));
			}
		}
	;
LOCAL_ATTR
	:	'$' ID {enclosingRule!=null && enclosingRule.getLocalAttributeScope($ID.text)!=null}?
		//{System.out.println("found \$localattr");}
		{
		StringTemplate st;
		AttributeScope scope = enclosingRule.getLocalAttributeScope($ID.text);
		if ( scope.isPredefinedRuleScope ) {
			st = template("rulePropertyRef_"+$ID.text);
			grammar.referenceRuleLabelPredefinedAttribute(enclosingRule.name);
			st.setAttribute("scope", enclosingRule.name);
			st.setAttribute("attr", $ID.text);
		}
		else if ( scope.isPredefinedLexerRuleScope ) {
			st = template("lexerRulePropertyRef_"+$ID.text);
			st.setAttribute("scope", enclosingRule.name);
			st.setAttribute("attr", $ID.text);
		}
		else if ( scope.isParameterScope ) {
			st = template("parameterAttributeRef");
			st.setAttribute("attr", scope.getAttribute($ID.text));
		}
		else {
			st = template("returnAttributeRef");
			st.setAttribute("ruleDescriptor", enclosingRule);
			st.setAttribute("attr", scope.getAttribute($ID.text));
		}
		}
	;

/**	$x::y	the only way to access the attributes within a dynamic scope
 * 			regardless of whether or not you are in the defining rule.
 *
 * 			scope Symbols { List names; }
 * 			r
 * 			scope {int i;}
 * 			scope Symbols;
 * 				:	{$r::i=3;} s {$Symbols::names;}
 * 				;
 * 			s	:	{$r::i; $Symbols::names;}
 * 				;
 */
SET_DYNAMIC_SCOPE_ATTR
	:	'$' x=ID '::' y=ID WS? '=' expr=ATTR_VALUE_EXPR ';'
						   {resolveDynamicScope($x.text)!=null &&
						     resolveDynamicScope($x.text).getAttribute($y.text)!=null}?
		//{System.out.println("found set \$scope::attr "+ $x.text + "::" + $y.text + " to " + $expr.text);}
		{
		AttributeScope scope = resolveDynamicScope($x.text);
		if ( scope!=null ) {
			StringTemplate st = template("scopeSetAttributeRef");
			st.setAttribute("scope", $x.text);
			st.setAttribute("attr",  scope.getAttribute($y.text));
			st.setAttribute("expr",  translateAction($expr.text));
		}
		else {
			// error: invalid dynamic attribute
		}
		}
	;

DYNAMIC_SCOPE_ATTR
	:	'$' x=ID '::' y=ID
						   {resolveDynamicScope($x.text)!=null &&
						     resolveDynamicScope($x.text).getAttribute($y.text)!=null}?
		//{System.out.println("found \$scope::attr "+ $x.text + "::" + $y.text);}
		{
		AttributeScope scope = resolveDynamicScope($x.text);
		if ( scope!=null ) {
			StringTemplate st = template("scopeAttributeRef");
			st.setAttribute("scope", $x.text);
			st.setAttribute("attr",  scope.getAttribute($y.text));
		}
		else {
			// error: invalid dynamic attribute
		}
		}
	;


ERROR_SCOPED_XY
	:	'$' x=ID '::' y=ID
		{
		chunks.add(getText());
		generator.issueInvalidScopeError($x.text,$y.text,
		                                 enclosingRule,actionToken,
		                                 outerAltNum);		
		}
	;
	
/**		To access deeper (than top of stack) scopes, use the notation:
 *
 * 		$x[-1]::y previous (just under top of stack)
 * 		$x[-i]::y top of stack - i where the '-' MUST BE PRESENT;
 * 				  i.e., i cannot simply be negative without the '-' sign!
 * 		$x[i]::y  absolute index i (0..size-1)
 * 		$x[0]::y  is the absolute 0 indexed element (bottom of the stack)
 */
DYNAMIC_NEGATIVE_INDEXED_SCOPE_ATTR
	:	'$' x=ID '[' '-' expr=SCOPE_INDEX_EXPR ']' '::' y=ID
		// {System.out.println("found \$scope[-...]::attr");}
		{
		StringTemplate st = template("scopeAttributeRef");
		st.setAttribute("scope",    $x.text);
		st.setAttribute("attr",     resolveDynamicScope($x.text).getAttribute($y.text));
		st.setAttribute("negIndex", $expr.text);
		}		
	;

DYNAMIC_ABSOLUTE_INDEXED_SCOPE_ATTR
	:	'$' x=ID '[' expr=SCOPE_INDEX_EXPR ']' '::' y=ID 
		// {System.out.println("found \$scope[...]::attr");}
		{
		StringTemplate st = template("scopeAttributeRef");
		st.setAttribute("scope", $x.text);
		st.setAttribute("attr",  resolveDynamicScope($x.text).getAttribute($y.text));
		st.setAttribute("index", $expr.text);
		}		
	;

fragment
SCOPE_INDEX_EXPR
	:	(~']')+
	;
	
/** $r		y is a rule's dynamic scope or a global shared scope.
 * 			Isolated $rulename is not allowed unless it has a dynamic scope *and*
 * 			there is no reference to rulename in the enclosing alternative,
 * 			which would be ambiguous.  See TestAttributes.testAmbiguousRuleRef()
 */
ISOLATED_DYNAMIC_SCOPE
	:	'$' ID {resolveDynamicScope($ID.text)!=null}?
		// {System.out.println("found isolated \$scope where scope is a dynamic scope");}
		{
		StringTemplate st = template("isolatedDynamicScopeRef");
		st.setAttribute("scope", $ID.text);
		}		
	;
	
// antlr.g then codegen.g does these first two currently.
// don't want to duplicate that code.

/** %foo(a={},b={},...) ctor */
TEMPLATE_INSTANCE
	:	'%' ID '(' ( WS? ARG (',' WS? ARG)* WS? )? ')'
		// {System.out.println("found \%foo(args)");}
		{
		String action = getText().substring(1,getText().length());
		String ruleName = "<outside-of-rule>";
		if ( enclosingRule!=null ) {
			ruleName = enclosingRule.name;
		}
		StringTemplate st =
			generator.translateTemplateConstructor(ruleName,
												   outerAltNum,
												   actionToken,
												   action);
		if ( st!=null ) {
			chunks.add(st);
		}
		}
	;

/** %({name-expr})(a={},...) indirect template ctor reference */
INDIRECT_TEMPLATE_INSTANCE
	:	'%' '(' ACTION ')' '(' ( WS? ARG (',' WS? ARG)* WS? )? ')'
		// {System.out.println("found \%({...})(args)");}
		{
		String action = getText().substring(1,getText().length());
		StringTemplate st =
			generator.translateTemplateConstructor(enclosingRule.name,
												   outerAltNum,
												   actionToken,
												   action);
		chunks.add(st);
		}
	;

fragment
ARG	:	ID '=' ACTION
	;

/**	%{expr}.y = z; template attribute y of StringTemplate-typed expr to z */
SET_EXPR_ATTRIBUTE
	:	'%' a=ACTION '.' ID WS? '=' expr=ATTR_VALUE_EXPR ';'
		// {System.out.println("found \%{expr}.y = z;");}
		{
		StringTemplate st = template("actionSetAttribute");
		String action = $a.text;
		action = action.substring(1,action.length()-1); // stuff inside {...}
		st.setAttribute("st", translateAction(action));
		st.setAttribute("attrName", $ID.text);
		st.setAttribute("expr", translateAction($expr.text));
		}
	;
	
/*    %x.y = z; set template attribute y of x (always set never get attr)
 *              to z [languages like python without ';' must still use the
 *              ';' which the code generator is free to remove during code gen]
 */
SET_ATTRIBUTE
	:	'%' x=ID '.' y=ID WS? '=' expr=ATTR_VALUE_EXPR ';'
		// {System.out.println("found \%x.y = z;");}
		{
		StringTemplate st = template("actionSetAttribute");
		st.setAttribute("st", $x.text);
		st.setAttribute("attrName", $y.text);
		st.setAttribute("expr", translateAction($expr.text));
		}
	;

/** Don't allow an = as first char to prevent $x == 3; kind of stuff. */
fragment
ATTR_VALUE_EXPR
	:	~'=' (~';')*
	;
	
/** %{string-expr} anonymous template from string expr */
TEMPLATE_EXPR
	:	'%' a=ACTION
		// {System.out.println("found \%{expr}");}
		{
		StringTemplate st = template("actionStringConstructor");
		String action = $a.text;
		action = action.substring(1,action.length()-1); // stuff inside {...}
		st.setAttribute("stringExpr", translateAction(action));
		}
	;
	
fragment
ACTION
	:	'{' (options {greedy=false;}:.)* '}'
	;
	
ESC :   '\\' '$' {chunks.add("\$");}
	|	'\\' '%' {chunks.add("\%");}
	|	'\\' ~('$'|'%') {chunks.add(getText());}
    ;       

ERROR_XY
	:	'$' x=ID '.' y=ID
		{
		chunks.add(getText());
		generator.issueInvalidAttributeError($x.text,$y.text,
		                                     enclosingRule,actionToken,
		                                     outerAltNum);
		}
	;
	
ERROR_X
	:	'$' x=ID
		{
		chunks.add(getText());
		generator.issueInvalidAttributeError($x.text,
		                                     enclosingRule,actionToken,
		                                     outerAltNum);
		}
	;
	
UNKNOWN_SYNTAX
	:	'$'
		{
		chunks.add(getText());
		// shouldn't need an error here.  Just accept \$ if it doesn't look like anything
		}
	|	'%' (ID|'.'|'('|')'|','|'{'|'}'|'"')*
		{
		chunks.add(getText());
		ErrorManager.grammarError(ErrorManager.MSG_INVALID_TEMPLATE_ACTION,
								  grammar,
								  actionToken,
								  getText());
		}
	;

TEXT:	~('$'|'%'|'\\')+ {chunks.add(getText());}
	;
	
fragment
ID  :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'_'|'0'..'9')*
    ;

fragment
INT :	'0'..'9'+
	;

fragment
WS	:	(' '|'\t'|'\n'|'\r')+
	;
