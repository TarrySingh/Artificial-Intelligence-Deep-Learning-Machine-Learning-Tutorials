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

import antlr.CommonToken;
import org.antlr.analysis.NFAState;
import org.antlr.analysis.LookaheadSet;
import org.antlr.codegen.CodeGenerator;

import java.util.*;
import org.antlr.grammar.v2.ANTLRParser;

/** Combine the info associated with a rule. */
public class Rule {
	public String name;
	public int index;
	public String modifier;
	public NFAState startState;
	public NFAState stopState;

	/** This rule's options */
	protected Map options;

	public static final Set legalOptions =
			new HashSet() {
                {
                    add("k"); add("greedy"); add("memoize");
                    add("backtrack"); add("strategy");
                }
            };

	/** The AST representing the whole rule */
	public GrammarAST tree;

	/** To which grammar does this belong? */
	public Grammar grammar;

	/** For convenience, track the argument def AST action node if any */
	public GrammarAST argActionAST;

	public GrammarAST EORNode;

	/** The set of all tokens reachable from the start state w/o leaving
	 *  via the accept state.  If it reaches the accept state, FIRST
	 *  includes EOR_TOKEN_TYPE.
	 */
	public LookaheadSet FIRST;

	/** The return values of a rule and predefined rule attributes */
	public AttributeScope returnScope;

	public AttributeScope parameterScope;

	/** the attributes defined with "scope {...}" inside a rule */
	public AttributeScope ruleScope;

	/** A list of scope names (String) used by this rule */
	public List useScopes;

    /** A list of all LabelElementPair attached to tokens like id=ID */
    public LinkedHashMap tokenLabels;

    /** A list of all LabelElementPair attached to tokens like x=. in tree grammar */
    public LinkedHashMap wildcardTreeLabels;

    /** A list of all LabelElementPair attached to tokens like x+=. in tree grammar */
    public LinkedHashMap wildcardTreeListLabels;

	/** A list of all LabelElementPair attached to single char literals like x='a' */
	public LinkedHashMap charLabels;

	/** A list of all LabelElementPair attached to rule references like f=field */
	public LinkedHashMap ruleLabels;

	/** A list of all Token list LabelElementPair like ids+=ID */
	public LinkedHashMap tokenListLabels;

	/** A list of all rule ref list LabelElementPair like ids+=expr */
	public LinkedHashMap ruleListLabels;

	/** All labels go in here (plus being split per the above lists) to
	 *  catch dup label and label type mismatches.
	 */
	protected Map<String, Grammar.LabelElementPair> labelNameSpace =
		new HashMap<String, Grammar.LabelElementPair>();

	/** Map a name to an action for this rule.  Currently init is only
	 *  one we use, but we can add more in future.
	 *  The code generator will use this to fill holes in the rule template.
	 *  I track the AST node for the action in case I need the line number
	 *  for errors.  A better name is probably namedActions, but I don't
	 *  want everyone to have to change their code gen templates now.
	 */
	protected Map<String, GrammarAST> actions =
		new HashMap<String, GrammarAST>();

	/** Track all executable actions other than named actions like @init.
	 *  Also tracks exception handlers, predicates, and rewrite rewrites.
	 *  We need to examine these actions before code generation so
	 *  that we can detect refs to $rule.attr etc...
	 */
	protected List<GrammarAST> inlineActions = new ArrayList<GrammarAST>();

	public int numberOfAlts;

	/** Each alt has a Map<tokenRefName,List<tokenRefAST>>; range 1..numberOfAlts.
	 *  So, if there are 3 ID refs in a rule's alt number 2, you'll have
	 *  altToTokenRef[2].get("ID").size()==3.  This is used to see if $ID is ok.
	 *  There must be only one ID reference in the alt for $ID to be ok in
	 *  an action--must be unique.
	 *
	 *  This also tracks '+' and "int" literal token references
	 *  (if not in LEXER).
	 *
	 *  Rewrite rules force tracking of all tokens.
	 */
	protected Map<String, List<GrammarAST>>[] altToTokenRefMap;

	/** Each alt has a Map<ruleRefName,List<ruleRefAST>>; range 1..numberOfAlts
	 *  So, if there are 3 expr refs in a rule's alt number 2, you'll have
	 *  altToRuleRef[2].get("expr").size()==3.  This is used to see if $expr is ok.
	 *  There must be only one expr reference in the alt for $expr to be ok in
	 *  an action--must be unique.
	 *
	 *  Rewrite rules force tracking of all rule result ASTs. 1..n
	 */
	protected Map<String, List<GrammarAST>>[] altToRuleRefMap;

	/** Track which alts have rewrite rules associated with them. 1..n */
	protected boolean[] altsWithRewrites;

	/** Do not generate start, stop etc... in a return value struct unless
	 *  somebody references $r.start somewhere.
	 */
	public boolean referencedPredefinedRuleAttributes = false;

	public boolean isSynPred = false;

	public boolean imported = false;

	public Rule(Grammar grammar,
				String ruleName,
				int ruleIndex,
				int numberOfAlts)
	{
		this.name = ruleName;
		this.index = ruleIndex;
		this.numberOfAlts = numberOfAlts;
		this.grammar = grammar;
		altToTokenRefMap = new Map[numberOfAlts+1];
		altToRuleRefMap = new Map[numberOfAlts+1];
		altsWithRewrites = new boolean[numberOfAlts+1];
		for (int alt=1; alt<=numberOfAlts; alt++) {
			altToTokenRefMap[alt] = new HashMap<String, List<GrammarAST>>();
			altToRuleRefMap[alt] = new HashMap<String, List<GrammarAST>>();
		}
	}

	public void defineLabel(antlr.Token label, GrammarAST elementRef, int type) {
		Grammar.LabelElementPair pair = grammar.new LabelElementPair(label,elementRef);
		pair.type = type;
		labelNameSpace.put(label.getText(), pair);
		switch ( type ) {
            case Grammar.TOKEN_LABEL :
                if ( tokenLabels==null ) tokenLabels = new LinkedHashMap();
                tokenLabels.put(label.getText(), pair);
                break;
            case Grammar.WILDCARD_TREE_LABEL :
                if ( wildcardTreeLabels==null ) wildcardTreeLabels = new LinkedHashMap();
                wildcardTreeLabels.put(label.getText(), pair);
                break;
            case Grammar.WILDCARD_TREE_LIST_LABEL :
                if ( wildcardTreeListLabels==null ) wildcardTreeListLabels = new LinkedHashMap();
                wildcardTreeListLabels.put(label.getText(), pair);
                break;
			case Grammar.RULE_LABEL :
				if ( ruleLabels==null ) ruleLabels = new LinkedHashMap();
				ruleLabels.put(label.getText(), pair);
				break;
			case Grammar.TOKEN_LIST_LABEL :
				if ( tokenListLabels==null ) tokenListLabels = new LinkedHashMap();
				tokenListLabels.put(label.getText(), pair);
				break;
			case Grammar.RULE_LIST_LABEL :
				if ( ruleListLabels==null ) ruleListLabels = new LinkedHashMap();
				ruleListLabels.put(label.getText(), pair);
				break;
			case Grammar.CHAR_LABEL :
				if ( charLabels==null ) charLabels = new LinkedHashMap();
				charLabels.put(label.getText(), pair);
				break;
		}
	}

	public Grammar.LabelElementPair getLabel(String name) {
		return (Grammar.LabelElementPair)labelNameSpace.get(name);
	}

	public Grammar.LabelElementPair getTokenLabel(String name) {
		Grammar.LabelElementPair pair = null;
		if ( tokenLabels!=null ) {
			return (Grammar.LabelElementPair)tokenLabels.get(name);
		}
		return pair;
	}

	public Map getRuleLabels() {
		return ruleLabels;
	}

	public Map getRuleListLabels() {
		return ruleListLabels;
	}

	public Grammar.LabelElementPair getRuleLabel(String name) {
		Grammar.LabelElementPair pair = null;
		if ( ruleLabels!=null ) {
			return (Grammar.LabelElementPair)ruleLabels.get(name);
		}
		return pair;
	}

	public Grammar.LabelElementPair getTokenListLabel(String name) {
		Grammar.LabelElementPair pair = null;
		if ( tokenListLabels!=null ) {
			return (Grammar.LabelElementPair)tokenListLabels.get(name);
		}
		return pair;
	}

	public Grammar.LabelElementPair getRuleListLabel(String name) {
		Grammar.LabelElementPair pair = null;
		if ( ruleListLabels!=null ) {
			return (Grammar.LabelElementPair)ruleListLabels.get(name);
		}
		return pair;
	}

	/** Track a token ID or literal like '+' and "void" as having been referenced
	 *  somewhere within the alts (not rewrite sections) of a rule.
	 *
	 *  This differs from Grammar.altReferencesTokenID(), which tracks all
	 *  token IDs to check for token IDs without corresponding lexer rules.
	 */
	public void trackTokenReferenceInAlt(GrammarAST refAST, int outerAltNum) {
		List refs = (List)altToTokenRefMap[outerAltNum].get(refAST.getText());
		if ( refs==null ) {
			refs = new ArrayList();
			altToTokenRefMap[outerAltNum].put(refAST.getText(), refs);
		}
		refs.add(refAST);
	}

	public List getTokenRefsInAlt(String ref, int outerAltNum) {
		if ( altToTokenRefMap[outerAltNum]!=null ) {
			List tokenRefASTs = (List)altToTokenRefMap[outerAltNum].get(ref);
			return tokenRefASTs;
		}
		return null;
	}

	public void trackRuleReferenceInAlt(GrammarAST refAST, int outerAltNum) {
		List refs = (List)altToRuleRefMap[outerAltNum].get(refAST.getText());
		if ( refs==null ) {
			refs = new ArrayList();
			altToRuleRefMap[outerAltNum].put(refAST.getText(), refs);
		}
		refs.add(refAST);
	}

	public List getRuleRefsInAlt(String ref, int outerAltNum) {
		if ( altToRuleRefMap[outerAltNum]!=null ) {
			List ruleRefASTs = (List)altToRuleRefMap[outerAltNum].get(ref);
			return ruleRefASTs;
		}
		return null;
	}

	public Set getTokenRefsInAlt(int altNum) {
		return altToTokenRefMap[altNum].keySet();
	}

	/** For use with rewrite rules, we must track all tokens matched on the
	 *  left-hand-side; so we need Lists.  This is a unique list of all
	 *  token types for which the rule needs a list of tokens.  This
	 *  is called from the rule template not directly by the code generator.
	 */
	public Set getAllTokenRefsInAltsWithRewrites() {
		String output = (String)grammar.getOption("output");
		Set tokens = new HashSet();
		if ( output==null || !output.equals("AST") ) {
			// return nothing if not generating trees; i.e., don't do for templates
			return tokens;
		}
		for (int i = 1; i <= numberOfAlts; i++) {
			if ( altsWithRewrites[i] ) {
				Map m = altToTokenRefMap[i];
				Set s = m.keySet();
				for (Iterator it = s.iterator(); it.hasNext();) {
					// convert token name like ID to ID, "void" to 31
					String tokenName = (String) it.next();
					int ttype = grammar.getTokenType(tokenName);
					String label = grammar.generator.getTokenTypeAsTargetLabel(ttype);
					tokens.add(label);
				}
			}
		}
		return tokens;
	}

	public Set getRuleRefsInAlt(int outerAltNum) {
		return altToRuleRefMap[outerAltNum].keySet();
	}

	/** For use with rewrite rules, we must track all rule AST results on the
	 *  left-hand-side; so we need Lists.  This is a unique list of all
	 *  rule results for which the rule needs a list of results.
	 */
	public Set getAllRuleRefsInAltsWithRewrites() {
		Set rules = new HashSet();
		for (int i = 1; i <= numberOfAlts; i++) {
			if ( altsWithRewrites[i] ) {
				Map m = altToRuleRefMap[i];
				rules.addAll(m.keySet());
			}
		}
		return rules;
	}

	public List<GrammarAST> getInlineActions() {
		return inlineActions;
	}

	public boolean hasRewrite(int i) {
		if ( i >= altsWithRewrites.length ) {
			ErrorManager.internalError("alt "+i+" exceeds number of "+name+
									   "'s alts ("+altsWithRewrites.length+")");
			return false;
		}
		return altsWithRewrites[i];
	}

	/** Track which rules have rewrite rules.  Pass in the ALT node
	 *  for the alt so we can check for problems when output=template,
	 *  rewrite=true, and grammar type is tree parser.
	 */
	public void trackAltsWithRewrites(GrammarAST altAST, int outerAltNum) {
		if ( grammar.type==Grammar.TREE_PARSER &&
			 grammar.buildTemplate() &&
			 grammar.getOption("rewrite")!=null &&
			 grammar.getOption("rewrite").equals("true")
			)
		{
			GrammarAST firstElementAST = (GrammarAST)altAST.getFirstChild();
			grammar.sanity.ensureAltIsSimpleNodeOrTree(altAST,
													   firstElementAST,
													   outerAltNum);
		}
		altsWithRewrites[outerAltNum] = true;
	}

	/** Return the scope containing name */
	public AttributeScope getAttributeScope(String name) {
		AttributeScope scope = getLocalAttributeScope(name);
		if ( scope!=null ) {
			return scope;
		}
		if ( ruleScope!=null && ruleScope.getAttribute(name)!=null ) {
			scope = ruleScope;
		}
		return scope;
	}

	/** Get the arg, return value, or predefined property for this rule */
	public AttributeScope getLocalAttributeScope(String name) {
		AttributeScope scope = null;
		if ( returnScope!=null && returnScope.getAttribute(name)!=null ) {
			scope = returnScope;
		}
		else if ( parameterScope!=null && parameterScope.getAttribute(name)!=null ) {
			scope = parameterScope;
		}
		else {
			AttributeScope rulePropertiesScope =
				RuleLabelScope.grammarTypeToRulePropertiesScope[grammar.type];
			if ( rulePropertiesScope.getAttribute(name)!=null ) {
				scope = rulePropertiesScope;
			}
		}
		return scope;
	}

	/** For references to tokens rather than by label such as $ID, we
	 *  need to get the existing label for the ID ref or create a new
	 *  one.
	 */
	public String getElementLabel(String refdSymbol,
								  int outerAltNum,
								  CodeGenerator generator)
	{
		GrammarAST uniqueRefAST;
		if ( grammar.type != Grammar.LEXER &&
			 Character.isUpperCase(refdSymbol.charAt(0)) )
		{
			// symbol is a token
			List tokenRefs = getTokenRefsInAlt(refdSymbol, outerAltNum);
			uniqueRefAST = (GrammarAST)tokenRefs.get(0);
		}
		else {
			// symbol is a rule
			List ruleRefs = getRuleRefsInAlt(refdSymbol, outerAltNum);
			uniqueRefAST = (GrammarAST)ruleRefs.get(0);
		}
		if ( uniqueRefAST.code==null ) {
			// no code?  must not have gen'd yet; forward ref
			return null;
		}
		String labelName = null;
		String existingLabelName =
			(String)uniqueRefAST.code.getAttribute("label");
		// reuse any label or list label if it exists
		if ( existingLabelName!=null ) {
			labelName = existingLabelName;
		}
		else {
			// else create new label
			labelName = generator.createUniqueLabel(refdSymbol);
			CommonToken label = new CommonToken(ANTLRParser.ID, labelName);
			if ( grammar.type != Grammar.LEXER &&
				 Character.isUpperCase(refdSymbol.charAt(0)) )
			{
				grammar.defineTokenRefLabel(name, label, uniqueRefAST);
			}
			else {
				grammar.defineRuleRefLabel(name, label, uniqueRefAST);
			}
			uniqueRefAST.code.setAttribute("label", labelName);
		}
		return labelName;
	}

	/** If a rule has no user-defined return values and nobody references
	 *  it's start/stop (predefined attributes), then there is no need to
	 *  define a struct; otherwise for now we assume a struct.  A rule also
	 *  has multiple return values if you are building trees or templates.
	 */
	public boolean getHasMultipleReturnValues() {
		return
			referencedPredefinedRuleAttributes || grammar.buildAST() ||
			grammar.buildTemplate() ||
			(returnScope!=null && returnScope.attributes.size()>1);
	}

	public boolean getHasSingleReturnValue() {
		return
			!(referencedPredefinedRuleAttributes || grammar.buildAST() ||
			  grammar.buildTemplate()) &&
									   (returnScope!=null && returnScope.attributes.size()==1);
	}

	public boolean getHasReturnValue() {
		return
			referencedPredefinedRuleAttributes || grammar.buildAST() ||
			grammar.buildTemplate() ||
			(returnScope!=null && returnScope.attributes.size()>0);
	}

	public String getSingleValueReturnType() {
		if ( returnScope!=null && returnScope.attributes.size()==1 ) {
			Collection retvalAttrs = returnScope.attributes.values();
			Object[] javaSucks = retvalAttrs.toArray();
			return ((Attribute)javaSucks[0]).type;
		}
		return null;
	}

	public String getSingleValueReturnName() {
		if ( returnScope!=null && returnScope.attributes.size()==1 ) {
			Collection retvalAttrs = returnScope.attributes.values();
			Object[] javaSucks = retvalAttrs.toArray();
			return ((Attribute)javaSucks[0]).name;
		}
		return null;
	}

	/** Given @scope::name {action} define it for this grammar.  Later,
	 *  the code generator will ask for the actions table.
	 */
	public void defineNamedAction(GrammarAST ampersandAST,
								  GrammarAST nameAST,
								  GrammarAST actionAST)
	{
		//System.out.println("rule @"+nameAST.getText()+"{"+actionAST.getText()+"}");
		String actionName = nameAST.getText();
		GrammarAST a = (GrammarAST)actions.get(actionName);
		if ( a!=null ) {
			ErrorManager.grammarError(
				ErrorManager.MSG_ACTION_REDEFINITION,grammar,
				nameAST.getToken(),nameAST.getText());
		}
		else {
			actions.put(actionName,actionAST);
		}
	}

	public void trackInlineAction(GrammarAST actionAST) {
		inlineActions.add(actionAST);
	}

	public Map<String, GrammarAST> getActions() {
		return actions;
	}

	public void setActions(Map<String, GrammarAST> actions) {
		this.actions = actions;
	}

	/** Save the option key/value pair and process it; return the key
	 *  or null if invalid option.
	 */
	public String setOption(String key, Object value, antlr.Token optionsStartToken) {
		if ( !legalOptions.contains(key) ) {
			ErrorManager.grammarError(ErrorManager.MSG_ILLEGAL_OPTION,
									  grammar,
									  optionsStartToken,
									  key);
			return null;
		}
		if ( options==null ) {
			options = new HashMap();
		}
        if ( key.equals("memoize") && value.toString().equals("true") ) {
            grammar.atLeastOneRuleMemoizes = true;
        }
        if ( key.equals("backtrack") && value.toString().equals("true") ) {
            grammar.composite.getRootGrammar().atLeastOneBacktrackOption = true;
        }
		if ( key.equals("k") ) {
			grammar.numberOfManualLookaheadOptions++;
		}
		 options.put(key, value);
		return key;
	}

	public void setOptions(Map options, antlr.Token optionsStartToken) {
		if ( options==null ) {
			this.options = null;
			return;
		}
		Set keys = options.keySet();
		for (Iterator it = keys.iterator(); it.hasNext();) {
			String optionName = (String) it.next();
			Object optionValue = options.get(optionName);
			String stored=setOption(optionName, optionValue, optionsStartToken);
			if ( stored==null ) {
				it.remove();
			}
		}
	}

	/** Used during grammar imports to see if sets of rules intersect... This
	 *  method and hashCode use the String name as the key for Rule objects.
	public boolean equals(Object other) {
		return this.name.equals(((Rule)other).name);
	}
	 */

	/** Used during grammar imports to see if sets of rules intersect...
	public int hashCode() {
		return name.hashCode();
	}
	 * */

	public String toString() { // used for testing
		return "["+grammar.name+"."+name+",index="+index+",line="+tree.getToken().getLine()+"]";
	}
}
