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

import org.antlr.analysis.Label;
import org.antlr.misc.Utils;

import java.util.*;

import org.antlr.grammar.v2.AssignTokenTypesWalker;

/** Move all of the functionality from assign.types.g grammar file. */
public class AssignTokenTypesBehavior extends AssignTokenTypesWalker {
	protected static final Integer UNASSIGNED = Utils.integer(-1);
	protected static final Integer UNASSIGNED_IN_PARSER_RULE = Utils.integer(-2);

	protected Map<String,Integer> stringLiterals = new LinkedHashMap();
	protected Map<String,Integer> tokens = new LinkedHashMap();
	protected Map<String,String> aliases = new LinkedHashMap();
	protected Map<String,String> aliasesReverseIndex = new HashMap<String,String>();

	/** Track actual lexer rule defs so we don't get repeated token defs in
	 *  generated lexer.
	 */
	protected Set<String> tokenRuleDefs = new HashSet();

    @Override
	protected void init(Grammar g) {
		this.grammar = g;
		currentRuleName = null;
		if ( stringAlias==null ) {
			// only init once; can't statically init since we need astFactory
			initASTPatterns();
		}
	}

	/** Track string literals (could be in tokens{} section) */
    @Override
	protected void trackString(GrammarAST t) {
		// if lexer, don't allow aliasing in tokens section
		if ( currentRuleName==null && grammar.type==Grammar.LEXER ) {
			ErrorManager.grammarError(ErrorManager.MSG_CANNOT_ALIAS_TOKENS_IN_LEXER,
									  grammar,
									  t.token,
									  t.getText());
			return;
		}
		// in a plain parser grammar rule, cannot reference literals
		// (unless defined previously via tokenVocab option)
		// don't warn until we hit root grammar as may be defined there.
		if ( grammar.getGrammarIsRoot() &&
			 grammar.type==Grammar.PARSER &&
			 grammar.getTokenType(t.getText())== Label.INVALID )
		{
			ErrorManager.grammarError(ErrorManager.MSG_LITERAL_NOT_ASSOCIATED_WITH_LEXER_RULE,
									  grammar,
									  t.token,
									  t.getText());
		}
		// Don't record literals for lexers, they are things to match not tokens
		if ( grammar.type==Grammar.LEXER ) {
			return;
		}
		// otherwise add literal to token types if referenced from parser rule
		// or in the tokens{} section
		if ( (currentRuleName==null ||
			  Character.isLowerCase(currentRuleName.charAt(0))) &&
																grammar.getTokenType(t.getText())==Label.INVALID )
		{
			stringLiterals.put(t.getText(), UNASSIGNED_IN_PARSER_RULE);
		}
	}

    @Override
	protected void trackToken(GrammarAST t) {
		// imported token names might exist, only add if new
		// Might have ';'=4 in vocab import and SEMI=';'. Avoid
		// setting to UNASSIGNED if we have loaded ';'/SEMI
		if ( grammar.getTokenType(t.getText())==Label.INVALID &&
			 tokens.get(t.getText())==null )
		{
			tokens.put(t.getText(), UNASSIGNED);
		}
	}

    @Override
	protected void trackTokenRule(GrammarAST t,
								  GrammarAST modifier,
								  GrammarAST block)
	{
		// imported token names might exist, only add if new
		if ( grammar.type==Grammar.LEXER || grammar.type==Grammar.COMBINED ) {
			if ( !Character.isUpperCase(t.getText().charAt(0)) ) {
				return;
			}
			if ( t.getText().equals(Grammar.ARTIFICIAL_TOKENS_RULENAME) ) {
				// don't add Tokens rule
				return;
			}

			// track all lexer rules so we can look for token refs w/o
			// associated lexer rules.
			grammar.composite.lexerRules.add(t.getText());

			int existing = grammar.getTokenType(t.getText());
			if ( existing==Label.INVALID ) {
				tokens.put(t.getText(), UNASSIGNED);
			}
			// look for "<TOKEN> : <literal> ;" pattern
			// (can have optional action last)
			if ( block.hasSameTreeStructure(charAlias) ||
				 block.hasSameTreeStructure(stringAlias) ||
				 block.hasSameTreeStructure(charAlias2) ||
				 block.hasSameTreeStructure(stringAlias2) )
			{
				tokenRuleDefs.add(t.getText());
				/*
			Grammar parent = grammar.composite.getDelegator(grammar);
			boolean importedByParserOrCombined =
				parent!=null &&
				(parent.type==Grammar.LEXER||parent.type==Grammar.PARSER);
				*/
				if ( grammar.type==Grammar.COMBINED || grammar.type==Grammar.LEXER ) {
					// only call this rule an alias if combined or lexer
					alias(t, (GrammarAST)block.getFirstChild().getFirstChild());
				}
			}
		}
		// else error
	}

    @Override
	protected void alias(GrammarAST t, GrammarAST s) {
		String tokenID = t.getText();
		String literal = s.getText();
		String prevAliasLiteralID = aliasesReverseIndex.get(literal);
		if ( prevAliasLiteralID!=null ) { // we've seen this literal before
			if ( tokenID.equals(prevAliasLiteralID) ) {
				// duplicate but identical alias; might be tokens {A='a'} and
				// lexer rule A : 'a' ;  Is ok, just return
				return;
			}

			// give error unless both are rules (ok if one is in tokens section)
			if ( !(tokenRuleDefs.contains(tokenID) && tokenRuleDefs.contains(prevAliasLiteralID)) )
			{
				// don't allow alias if A='a' in tokens section and B : 'a'; is rule.
				// Allow if both are rules.  Will get DFA nondeterminism error later.
				ErrorManager.grammarError(ErrorManager.MSG_TOKEN_ALIAS_CONFLICT,
										  grammar,
										  t.token,
										  tokenID+"="+literal,
										  prevAliasLiteralID);
			}
			return; // don't do the alias
		}
		int existingLiteralType = grammar.getTokenType(literal);
		if ( existingLiteralType !=Label.INVALID ) {
			// we've seen this before from a tokenVocab most likely
			// don't assign a new token type; use existingLiteralType.
			tokens.put(tokenID, existingLiteralType);
		}
		String prevAliasTokenID = aliases.get(tokenID);
		if ( prevAliasTokenID!=null ) {
			ErrorManager.grammarError(ErrorManager.MSG_TOKEN_ALIAS_REASSIGNMENT,
									  grammar,
									  t.token,
									  tokenID+"="+literal,
									  prevAliasTokenID);
			return; // don't do the alias
		}
		aliases.put(tokenID, literal);
		aliasesReverseIndex.put(literal, tokenID);
	}

    @Override
	public void defineTokens(Grammar root) {
/*
	System.out.println("stringLiterals="+stringLiterals);
	System.out.println("tokens="+tokens);
	System.out.println("aliases="+aliases);
	System.out.println("aliasesReverseIndex="+aliasesReverseIndex);
*/

		assignTokenIDTypes(root);

		aliasTokenIDsAndLiterals(root);

		assignStringTypes(root);

/*
	System.out.println("stringLiterals="+stringLiterals);
	System.out.println("tokens="+tokens);
	System.out.println("aliases="+aliases);
*/
		defineTokenNamesAndLiteralsInGrammar(root);
	}

/*
protected void defineStringLiteralsFromDelegates() {
	 if ( grammar.getGrammarIsMaster() && grammar.type==Grammar.COMBINED ) {
		 List<Grammar> delegates = grammar.getDelegates();
		 System.out.println("delegates in master combined: "+delegates);
		 for (int i = 0; i < delegates.size(); i++) {
			 Grammar d = (Grammar) delegates.get(i);
			 Set<String> literals = d.getStringLiterals();
			 for (Iterator it = literals.iterator(); it.hasNext();) {
				 String literal = (String) it.next();
				 System.out.println("literal "+literal);
				 int ttype = grammar.getTokenType(literal);
				 grammar.defineLexerRuleForStringLiteral(literal, ttype);
			 }
		 }
	 }
}
*/

    @Override
	protected void assignStringTypes(Grammar root) {
		// walk string literals assigning types to unassigned ones
		Set s = stringLiterals.keySet();
		for (Iterator it = s.iterator(); it.hasNext();) {
			String lit = (String) it.next();
			Integer oldTypeI = (Integer)stringLiterals.get(lit);
			int oldType = oldTypeI.intValue();
			if ( oldType<Label.MIN_TOKEN_TYPE ) {
				Integer typeI = Utils.integer(root.getNewTokenType());
				stringLiterals.put(lit, typeI);
				// if string referenced in combined grammar parser rule,
				// automatically define in the generated lexer
				root.defineLexerRuleForStringLiteral(lit, typeI.intValue());
			}
		}
	}

    @Override
	protected void aliasTokenIDsAndLiterals(Grammar root) {
		if ( root.type==Grammar.LEXER ) {
			return; // strings/chars are never token types in LEXER
		}
		// walk aliases if any and assign types to aliased literals if literal
		// was referenced
		Set s = aliases.keySet();
		for (Iterator it = s.iterator(); it.hasNext();) {
			String tokenID = (String) it.next();
			String literal = (String)aliases.get(tokenID);
			if ( literal.charAt(0)=='\'' && stringLiterals.get(literal)!=null ) {
				stringLiterals.put(literal, tokens.get(tokenID));
				// an alias still means you need a lexer rule for it
				Integer typeI = (Integer)tokens.get(tokenID);
				if ( !tokenRuleDefs.contains(tokenID) ) {
					root.defineLexerRuleForAliasedStringLiteral(tokenID, literal, typeI.intValue());
				}
			}
		}
	}

    @Override
	protected void assignTokenIDTypes(Grammar root) {
		// walk token names, assigning values if unassigned
		Set s = tokens.keySet();
		for (Iterator it = s.iterator(); it.hasNext();) {
			String tokenID = (String) it.next();
			if ( tokens.get(tokenID)==UNASSIGNED ) {
				tokens.put(tokenID, Utils.integer(root.getNewTokenType()));
			}
		}
	}

    @Override
	protected void defineTokenNamesAndLiteralsInGrammar(Grammar root) {
		Set s = tokens.keySet();
		for (Iterator it = s.iterator(); it.hasNext();) {
			String tokenID = (String) it.next();
			int ttype = ((Integer)tokens.get(tokenID)).intValue();
			root.defineToken(tokenID, ttype);
		}
		s = stringLiterals.keySet();
		for (Iterator it = s.iterator(); it.hasNext();) {
			String lit = (String) it.next();
			int ttype = ((Integer)stringLiterals.get(lit)).intValue();
			root.defineToken(lit, ttype);
		}
	}

}
