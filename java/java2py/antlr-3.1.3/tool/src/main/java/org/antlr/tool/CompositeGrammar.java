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
*/package org.antlr.tool;

import antlr.RecognitionException;
import org.antlr.analysis.Label;
import org.antlr.analysis.NFAState;
import org.antlr.misc.Utils;

import java.util.*;
import org.antlr.grammar.v2.AssignTokenTypesWalker;

/** A tree of component (delegate) grammars.
 *
 *  Rules defined in delegates are "inherited" like multi-inheritance
 *  so you can override them.  All token types must be consistent across
 *  rules from all delegate grammars, so they must be stored here in one
 *  central place.
 *
 *  We have to start out assuming a composite grammar situation as we can't
 *  look into the grammar files a priori to see if there is a delegate
 *  statement.  Because of this, and to avoid duplicating token type tracking
 *  in each grammar, even single noncomposite grammars use one of these objects
 *  to track token types.
 */
public class CompositeGrammar {
	public static final int MIN_RULE_INDEX = 1;
	
	public CompositeGrammarTree delegateGrammarTreeRoot;

	/** Used during getRuleReferenceClosure to detect computation cycles */
	protected Set<NFAState> refClosureBusy = new HashSet<NFAState>();

	/** Used to assign state numbers; all grammars in composite share common
	 *  NFA space.  This NFA tracks state numbers number to state mapping.
	 */
	public int stateCounter = 0;

	/** The NFA states in the NFA built from rules across grammars in composite.
	 *  Maps state number to NFAState object.
	 *  This is a Vector instead of a List because I need to be able to grow
	 *  this properly.  After talking to Josh Bloch, Collections guy at Sun,
	 *  I decided this was easiest solution.
	 */
	protected Vector<NFAState> numberToStateList = new Vector<NFAState>(1000);

	/** Token names and literal tokens like "void" are uniquely indexed.
	 *  with -1 implying EOF.  Characters are different; they go from
	 *  -1 (EOF) to \uFFFE.  For example, 0 could be a binary byte you
	 *  want to lexer.  Labels of DFA/NFA transitions can be both tokens
	 *  and characters.  I use negative numbers for bookkeeping labels
	 *  like EPSILON. Char/String literals and token types overlap in the same
	 *  space, however.
	 */
	protected int maxTokenType = Label.MIN_TOKEN_TYPE-1;

	/** Map token like ID (but not literals like "while") to its token type */
	public Map tokenIDToTypeMap = new HashMap();

	/** Map token literals like "while" to its token type.  It may be that
	 *  WHILE="while"=35, in which case both tokenIDToTypeMap and this
	 *  field will have entries both mapped to 35.
	 */
	public Map<String, Integer> stringLiteralToTypeMap = new HashMap<String, Integer>();
	/** Reverse index for stringLiteralToTypeMap */
	public Vector<String> typeToStringLiteralList = new Vector<String>();

	/** Map a token type to its token name.
	 *  Must subtract MIN_TOKEN_TYPE from index.
	 */
	public Vector<String> typeToTokenList = new Vector<String>();

	/** If combined or lexer grammar, track the rules.
	 * 	Track lexer rules so we can warn about undefined tokens.
	 *  This is combined set of lexer rules from all lexer grammars
	 *  seen in all imports.
	 */
	protected Set<String> lexerRules = new HashSet<String>();

	/** Rules are uniquely labeled from 1..n among all grammars */
	protected int ruleIndex = MIN_RULE_INDEX;

	/** Map a rule index to its name; use a Vector on purpose as new
	 *  collections stuff won't let me setSize and make it grow.  :(
	 *  I need a specific guaranteed index, which the Collections stuff
	 *  won't let me have.
	 */
	protected Vector<Rule> ruleIndexToRuleList = new Vector<Rule>();

	public boolean watchNFAConversion = false;

	protected void initTokenSymbolTables() {
		// the faux token types take first NUM_FAUX_LABELS positions
		// then we must have room for the predefined runtime token types
		// like DOWN/UP used for tree parsing.
		typeToTokenList.setSize(Label.NUM_FAUX_LABELS+Label.MIN_TOKEN_TYPE-1);
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.INVALID, "<INVALID>");
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.EOT, "<EOT>");
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.SEMPRED, "<SEMPRED>");
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.SET, "<SET>");
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.EPSILON, Label.EPSILON_STR);
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.EOF, "EOF");
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.EOR_TOKEN_TYPE-1, "<EOR>");
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.DOWN-1, "DOWN");
		typeToTokenList.set(Label.NUM_FAUX_LABELS+Label.UP-1, "UP");
		tokenIDToTypeMap.put("<INVALID>", Utils.integer(Label.INVALID));
		tokenIDToTypeMap.put("<EOT>", Utils.integer(Label.EOT));
		tokenIDToTypeMap.put("<SEMPRED>", Utils.integer(Label.SEMPRED));
		tokenIDToTypeMap.put("<SET>", Utils.integer(Label.SET));
		tokenIDToTypeMap.put("<EPSILON>", Utils.integer(Label.EPSILON));
		tokenIDToTypeMap.put("EOF", Utils.integer(Label.EOF));
		tokenIDToTypeMap.put("<EOR>", Utils.integer(Label.EOR_TOKEN_TYPE));
		tokenIDToTypeMap.put("DOWN", Utils.integer(Label.DOWN));
		tokenIDToTypeMap.put("UP", Utils.integer(Label.UP));
	}

	public CompositeGrammar() {
		initTokenSymbolTables();
	}

	public CompositeGrammar(Grammar g) {
		this();
		setDelegationRoot(g);
	}

	public void setDelegationRoot(Grammar root) {
		delegateGrammarTreeRoot = new CompositeGrammarTree(root);
		root.compositeTreeNode = delegateGrammarTreeRoot;
	}

	public Rule getRule(String ruleName) {
		return delegateGrammarTreeRoot.getRule(ruleName);
	}

	public Object getOption(String key) {
		return delegateGrammarTreeRoot.getOption(key);
	}

	/** Add delegate grammar as child of delegator */
	public void addGrammar(Grammar delegator, Grammar delegate) {
		if ( delegator.compositeTreeNode==null ) {
			delegator.compositeTreeNode = new CompositeGrammarTree(delegator);
		}
		delegator.compositeTreeNode.addChild(new CompositeGrammarTree(delegate));

		/*// find delegator in tree so we can add a child to it
		CompositeGrammarTree t = delegateGrammarTreeRoot.findNode(delegator);
		t.addChild();
		*/
		// make sure new grammar shares this composite
		delegate.composite = this;
	}

	/** Get parent of this grammar */
	public Grammar getDelegator(Grammar g) {
		CompositeGrammarTree me = delegateGrammarTreeRoot.findNode(g);
		if ( me==null ) {
			return null; // not found
		}
		if ( me.parent!=null ) {
			return me.parent.grammar;
		}
		return null;
	}

	/** Get list of all delegates from all grammars in the delegate subtree of g.
	 *  The grammars are in delegation tree preorder.  Don't include g itself
	 *  in list as it is not a delegate of itself.
	 */
	public List<Grammar> getDelegates(Grammar g) {
		CompositeGrammarTree t = delegateGrammarTreeRoot.findNode(g);
		if ( t==null ) {
			return null; // no delegates
		}
		List<Grammar> grammars = t.getPostOrderedGrammarList();
		grammars.remove(grammars.size()-1); // remove g (last one)
		return grammars;
	}

	public List<Grammar> getDirectDelegates(Grammar g) {
		CompositeGrammarTree t = delegateGrammarTreeRoot.findNode(g);
		List<CompositeGrammarTree> children = t.children;
		if ( children==null ) {
			return null;
		}
		List<Grammar> grammars = new ArrayList();
		for (int i = 0; children!=null && i < children.size(); i++) {
			CompositeGrammarTree child = (CompositeGrammarTree) children.get(i);
			grammars.add(child.grammar);
		}
		return grammars;
	}

	/** Get delegates below direct delegates of g */
	public List<Grammar> getIndirectDelegates(Grammar g) {
		List<Grammar> direct = getDirectDelegates(g);
		List<Grammar> delegates = getDelegates(g);
		delegates.removeAll(direct);
		return delegates;
	}

	/** Return list of delegate grammars from root down to g.
	 *  Order is root, ..., g.parent.  (g not included).
	 */
	public List<Grammar> getDelegators(Grammar g) {
		if ( g==delegateGrammarTreeRoot.grammar ) {
			return null;
		}
		List<Grammar> grammars = new ArrayList();
		CompositeGrammarTree t = delegateGrammarTreeRoot.findNode(g);
		// walk backwards to root, collecting grammars
		CompositeGrammarTree p = t.parent;
		while ( p!=null ) {
			grammars.add(0, p.grammar); // add to head so in order later
			p = p.parent;
		}
		return grammars;
	}

	/** Get set of rules for grammar g that need to have manual delegation
	 *  methods.  This is the list of rules collected from all direct/indirect
	 *  delegates minus rules overridden in grammar g.
	 *
	 *  This returns null except for the delegate root because it is the only
	 *  one that has to have a complete grammar rule interface.  The delegates
	 *  should not be instantiated directly for use as parsers (you can create
	 *  them to pass to the root parser's ctor as arguments).
	 */
	public Set<Rule> getDelegatedRules(Grammar g) {
		if ( g!=delegateGrammarTreeRoot.grammar ) {
			return null;
		}
		Set<Rule> rules = getAllImportedRules(g);
		for (Iterator it = rules.iterator(); it.hasNext();) {
			Rule r = (Rule) it.next();
			Rule localRule = g.getLocallyDefinedRule(r.name);
			// if locally defined or it's not local but synpred, don't make
			// a delegation method
			if ( localRule!=null || r.isSynPred ) {
				it.remove(); // kill overridden rules
			}
		}
		return rules;
	}

	/** Get all rule definitions from all direct/indirect delegate grammars
	 *  of g.
	 */
	public Set<Rule> getAllImportedRules(Grammar g) {
		Set<String> ruleNames = new HashSet();
		Set<Rule> rules = new HashSet();
		CompositeGrammarTree subtreeRoot = delegateGrammarTreeRoot.findNode(g);
		List<Grammar> grammars = subtreeRoot.getPostOrderedGrammarList();
		// walk all grammars
		for (int i = 0; i < grammars.size(); i++) {
			Grammar delegate = (org.antlr.tool.Grammar) grammars.get(i);
			// for each rule in delegate, add to rules if no rule with that
			// name as been seen.  (can't use removeAll; wrong hashcode/equals on Rule)
			for (Iterator it = delegate.getRules().iterator(); it.hasNext();) {
				Rule r = (Rule)it.next();
				if ( !ruleNames.contains(r.name) ) {
					ruleNames.add(r.name); // track that we've seen this
					rules.add(r);
				}
			}
		}
		return rules;
	}

	public Grammar getRootGrammar() {
		if ( delegateGrammarTreeRoot==null ) {
			return null;
		}
		return delegateGrammarTreeRoot.grammar;
	}

	public Grammar getGrammar(String grammarName) {
		CompositeGrammarTree t = delegateGrammarTreeRoot.findNode(grammarName);
		if ( t!=null ) {
			return t.grammar;
		}
		return null;
	}

	// NFA spans multiple grammars, must handle here

	public int getNewNFAStateNumber() {
		return stateCounter++;
	}

	public void addState(NFAState state) {
		numberToStateList.setSize(state.stateNumber+1); // make sure we have room
		numberToStateList.set(state.stateNumber, state);
	}

	public NFAState getState(int s) {
		return (NFAState)numberToStateList.get(s);
	}

	public void assignTokenTypes() throws antlr.RecognitionException {
		// ASSIGN TOKEN TYPES for all delegates (same walker)
		//System.out.println("### assign types");
		AssignTokenTypesWalker ttypesWalker = new AssignTokenTypesBehavior();
		ttypesWalker.setASTNodeClass("org.antlr.tool.GrammarAST");
		List<Grammar> grammars = delegateGrammarTreeRoot.getPostOrderedGrammarList();
		for (int i = 0; grammars!=null && i < grammars.size(); i++) {
			Grammar g = (Grammar)grammars.get(i);
			try {
				//System.out.println("    walking "+g.name);
				ttypesWalker.grammar(g.getGrammarTree(), g);
			}
			catch (RecognitionException re) {
				ErrorManager.error(ErrorManager.MSG_BAD_AST_STRUCTURE,
								   re);
			}
		}
		// the walker has filled literals, tokens, and alias tables.
		// now tell it to define them in the root grammar
		ttypesWalker.defineTokens(delegateGrammarTreeRoot.grammar);
	}

	public void defineGrammarSymbols() {
		delegateGrammarTreeRoot.trimLexerImportsIntoCombined();
		List<Grammar> grammars = delegateGrammarTreeRoot.getPostOrderedGrammarList();
		for (int i = 0; grammars!=null && i < grammars.size(); i++) {
			Grammar g = (Grammar)grammars.get(i);
			g.defineGrammarSymbols();
		}
		for (int i = 0; grammars!=null && i < grammars.size(); i++) {
			Grammar g = (Grammar)grammars.get(i);
			g.checkNameSpaceAndActions();
		}
		minimizeRuleSet();
	}

	public void createNFAs() {
		if ( ErrorManager.doNotAttemptAnalysis() ) {
			return;
		}
		List<Grammar> grammars = delegateGrammarTreeRoot.getPostOrderedGrammarList();
		List<String> names = new ArrayList<String>();
		for (int i = 0; i < grammars.size(); i++) {
			Grammar g = (Grammar) grammars.get(i);
			names.add(g.name);
		}
		//System.out.println("### createNFAs for composite; grammars: "+names);
		for (int i = 0; grammars!=null && i < grammars.size(); i++) {
			Grammar g = (Grammar)grammars.get(i);
			g.createRuleStartAndStopNFAStates();
		}
		for (int i = 0; grammars!=null && i < grammars.size(); i++) {
			Grammar g = (Grammar)grammars.get(i);
			g.buildNFA();
		}
	}

	public void minimizeRuleSet() {
		Set<String> ruleDefs = new HashSet<String>();
		_minimizeRuleSet(ruleDefs, delegateGrammarTreeRoot);
	}

	public void _minimizeRuleSet(Set<String> ruleDefs,
								 CompositeGrammarTree p) {
		Set<String> localRuleDefs = new HashSet<String>();
		Set<String> overrides = new HashSet<String>();
		// compute set of non-overridden rules for this delegate
		for (Rule r : p.grammar.getRules()) {
			if ( !ruleDefs.contains(r.name) ) {
				localRuleDefs.add(r.name);
			}
			else if ( !r.name.equals(Grammar.ARTIFICIAL_TOKENS_RULENAME) ) {
				// record any overridden rule 'cept tokens rule
				overrides.add(r.name);
			}
		}
		//System.out.println("rule defs for "+p.grammar.name+": "+localRuleDefs);
		//System.out.println("overridden rule for "+p.grammar.name+": "+overrides);
		p.grammar.overriddenRules = overrides;

		// make set of all rules defined thus far walking delegation tree.
		// the same rule in two delegates resolves in favor of first found
		// in tree therefore second must not be included
		ruleDefs.addAll(localRuleDefs);

		// pass larger set of defined rules to delegates
		if ( p.children!=null ) {
			for (CompositeGrammarTree delegate : p.children) {
				_minimizeRuleSet(ruleDefs, delegate);
			}
		}
	}

	/*
	public void minimizeRuleSet() {
		Set<Rule> refs = _minimizeRuleSet(delegateGrammarTreeRoot);
		System.out.println("all rule refs: "+refs);
	}

	public Set<Rule> _minimizeRuleSet(CompositeGrammarTree p) {
		Set<Rule> refs = new HashSet<Rule>();
		for (GrammarAST refAST : p.grammar.ruleRefs) {
			System.out.println("ref "+refAST.getText()+": "+refAST.NFAStartState+
							   " enclosing rule: "+refAST.NFAStartState.enclosingRule+
							   " invoking rule: "+((NFAState)refAST.NFAStartState.transition[0].target).enclosingRule);
			refs.add(((NFAState)refAST.NFAStartState.transition[0].target).enclosingRule);
		}

		if ( p.children!=null ) {
			for (CompositeGrammarTree delegate : p.children) {
				Set<Rule> delegateRuleRefs = _minimizeRuleSet(delegate);
				refs.addAll(delegateRuleRefs);
			}
		}

		return refs;
	}
	*/

	/*
	public void oldminimizeRuleSet() {
		// first walk to remove all overridden rules
		Set<String> ruleDefs = new HashSet<String>();
		Set<String> ruleRefs = new HashSet<String>();
		for (GrammarAST refAST : delegateGrammarTreeRoot.grammar.ruleRefs) {
			String rname = refAST.getText();
			ruleRefs.add(rname);
		}
		_minimizeRuleSet(ruleDefs,
						 ruleRefs,
						 delegateGrammarTreeRoot);
		System.out.println("overall rule defs: "+ruleDefs);
	}

	public void _minimizeRuleSet(Set<String> ruleDefs,
								 Set<String> ruleRefs,
								 CompositeGrammarTree p) {
		Set<String> localRuleDefs = new HashSet<String>();
		for (Rule r : p.grammar.getRules()) {
			if ( !ruleDefs.contains(r.name) ) {
				localRuleDefs.add(r.name);
				ruleDefs.add(r.name);
			}
		}
		System.out.println("rule defs for "+p.grammar.name+": "+localRuleDefs);

		// remove locally-defined rules not in ref set
		// find intersection of local rules and references from delegator
		// that is set of rules needed by delegator
		Set<String> localRuleDefsSatisfyingRefsFromBelow = new HashSet<String>();
		for (String r : ruleRefs) {
			if ( localRuleDefs.contains(r) ) {
				localRuleDefsSatisfyingRefsFromBelow.add(r);
			}
		}

		// now get list of refs from localRuleDefsSatisfyingRefsFromBelow.
		// Those rules are also allowed in this delegate
		for (GrammarAST refAST : p.grammar.ruleRefs) {
			if ( localRuleDefsSatisfyingRefsFromBelow.contains(refAST.enclosingRuleName) ) {
				// found rule ref within needed rule
			}
		}

		// remove rule refs not in the new rule def set

		// walk all children, adding rules not already defined
		if ( p.children!=null ) {
			for (CompositeGrammarTree delegate : p.children) {
				_minimizeRuleSet(ruleDefs, ruleRefs, delegate);
			}
		}
	}
	*/

	/*
	public void trackNFAStatesThatHaveLabeledEdge(Label label,
												  NFAState stateWithLabeledEdge)
	{
		Set<NFAState> states = typeToNFAStatesWithEdgeOfTypeMap.get(label);
		if ( states==null ) {
			states = new HashSet<NFAState>();
			typeToNFAStatesWithEdgeOfTypeMap.put(label, states);
		}
		states.add(stateWithLabeledEdge);
	}

	public Map<Label, Set<NFAState>> getTypeToNFAStatesWithEdgeOfTypeMap() {
		return typeToNFAStatesWithEdgeOfTypeMap;
	}

	public Set<NFAState> getStatesWithEdge(Label label) {
		return typeToNFAStatesWithEdgeOfTypeMap.get(label);
	}
*/
}
