header {
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
	package org.antlr.grammar.v2;
	import java.util.*;
	import org.antlr.analysis.*;
	import org.antlr.misc.*;
	import java.io.*;
    import org.antlr.tool.*;
}

/** [Warning: TJP says that this is probably out of date as of 11/19/2005,
 *   but since it's probably still useful, I'll leave in.  Don't have energy
 *   to update at the moment.]
 *
 *  Compute the token types for all literals and rules etc..  There are
 *  a few different cases to consider for grammar types and a few situations
 *  within.
 *
 *  CASE 1 : pure parser grammar
 *	a) Any reference to a token gets a token type.
 *  b) The tokens section may alias a token name to a string or char
 *
 *  CASE 2 : pure lexer grammar
 *  a) Import token vocabulary if available. Set token types for any new tokens
 *     to values above last imported token type
 *  b) token rule definitions get token types if not already defined
 *  c) literals do NOT get token types
 *
 *  CASE 3 : merged parser / lexer grammar
 *	a) Any char or string literal gets a token type in a parser rule
 *  b) Any reference to a token gets a token type if not referencing
 *     a fragment lexer rule
 *  c) The tokens section may alias a token name to a string or char
 *     which must add a rule to the lexer
 *  d) token rule definitions get token types if not already defined
 *  e) token rule definitions may also alias a token name to a literal.
 *     E.g., Rule 'FOR : "for";' will alias FOR to "for" in the sense that
 *     references to either in the parser grammar will yield the token type
 *
 *  What this pass does:
 *
 *  0. Collects basic info about the grammar like grammar name and type;
 *     Oh, I have go get the options in case they affect the token types.
 *     E.g., tokenVocab option.
 *     Imports any token vocab name/type pairs into a local hashtable.
 *  1. Finds a list of all literals and token names.
 *  2. Finds a list of all token name rule definitions;
 *     no token rules implies pure parser.
 *  3. Finds a list of all simple token rule defs of form "<NAME> : <literal>;"
 *     and aliases them.
 *  4. Walks token names table and assign types to any unassigned
 *  5. Walks aliases and assign types to referenced literals
 *  6. Walks literals, assigning types if untyped
 *  4. Informs the Grammar object of the type definitions such as:
 *     g.defineToken(<charliteral>, ttype);
 *     g.defineToken(<stringliteral>, ttype);
 *     g.defineToken(<tokenID>, ttype);
 *     where some of the ttype values will be the same for aliases tokens.
 */
class AssignTokenTypesWalker extends TreeParser;

options {
	importVocab = ANTLR;
	ASTLabelType = "GrammarAST";
    codeGenBitsetTestThreshold=999;
}

{
    public void reportError(RecognitionException ex) {
		Token token = null;
		if ( ex instanceof MismatchedTokenException ) {
			token = ((MismatchedTokenException)ex).token;
		}
		else if ( ex instanceof NoViableAltException ) {
			token = ((NoViableAltException)ex).token;
		}
        ErrorManager.syntaxError(
            ErrorManager.MSG_SYNTAX_ERROR,
            grammar,
            token,
            "assign.types: "+ex.toString(),
            ex);
    }


protected Grammar grammar;
protected String currentRuleName;

protected static GrammarAST stringAlias;
protected static GrammarAST charAlias;
protected static GrammarAST stringAlias2;
protected static GrammarAST charAlias2;

protected void initASTPatterns() {
	stringAlias =
		#(#[BLOCK], #(#[ALT], #[STRING_LITERAL], #[EOA]), #[EOB]);
	charAlias =
		#(#[BLOCK], #(#[ALT], #[CHAR_LITERAL], #[EOA]), #[EOB]);
	stringAlias2 =
		#(#[BLOCK], #(#[ALT], #[STRING_LITERAL], #[ACTION], #[EOA]),#[EOB]);
	charAlias2 =
		#(#[BLOCK], #(#[ALT], #[CHAR_LITERAL], #[ACTION], #[EOA]), #[EOB]);
}

// Behavior moved to AssignTokenTypesBehavior
protected void trackString(GrammarAST t) {;}
protected void trackToken(GrammarAST t) {;}
protected void trackTokenRule(GrammarAST t, GrammarAST modifier, GrammarAST block) {;}
protected void alias(GrammarAST t, GrammarAST s) {;}
public    void defineTokens(Grammar root) {;}
protected void defineStringLiteralsFromDelegates() {;}
protected void assignStringTypes(Grammar root) {;}
protected void aliasTokenIDsAndLiterals(Grammar root) {;}
protected void assignTokenIDTypes(Grammar root) {;}
protected void defineTokenNamesAndLiteralsInGrammar(Grammar root) {;}
protected void init(Grammar root) {;}
}

grammar[Grammar g]
{
	init(g);
}
    :   ( #( LEXER_GRAMMAR 	  grammarSpec )
	    | #( PARSER_GRAMMAR   grammarSpec )
	    | #( TREE_GRAMMAR     grammarSpec )
	    | #( COMBINED_GRAMMAR grammarSpec )
	    )
    ;

grammarSpec
{Map opts=null;}
	:	id:ID
		(cmt:DOC_COMMENT)?
		(optionsSpec)?
        (delegateGrammars)?
        (tokensSpec)?
        (attrScope)*
        (AMPERSAND)* // skip actions
        rules
	;

attrScope
	:	#( "scope" ID ACTION )
	;

optionsSpec returns [Map opts=new HashMap()]
    :   #( OPTIONS (option[opts])+ )
    ;

option[Map opts]
{
    String key=null;
    Object value=null;
}
    :   #( ASSIGN id:ID {key=#id.getText();} value=optionValue )
        {
        opts.put(key,value);
        // check for grammar-level option to import vocabulary
        if ( currentRuleName==null && key.equals("tokenVocab") ) {
            grammar.importTokenVocabulary(#id,(String)value);
        }
        }
    ;

optionValue returns [Object value=null]
    :   id:ID			 {value = #id.getText();}
    |   s:STRING_LITERAL {value = #s.getText();}
    |   c:CHAR_LITERAL   {value = #c.getText();}
    |   i:INT            {value = new Integer(#i.getText());}
//  |   cs:charSet       {value = #cs;} // return set AST in this case
    ;

charSet
	:   #( CHARSET charSetElement )
	;

charSetElement
	:   c:CHAR_LITERAL
	|   #( OR c1:CHAR_LITERAL c2:CHAR_LITERAL )
	|   #( RANGE c3:CHAR_LITERAL c4:CHAR_LITERAL )
	;

delegateGrammars
	:	#( "import"
            (   #(ASSIGN ID ID)
            |   ID
            )+
        )
	;

tokensSpec
	:	#( TOKENS ( tokenSpec )+ )
	;

tokenSpec
	:	t:TOKEN_REF           {trackToken(t);}
	|	#( ASSIGN
		   t2:TOKEN_REF       {trackToken(t2);}
		   ( s:STRING_LITERAL {trackString(s); alias(t2,s);}
		   | c:CHAR_LITERAL   {trackString(c); alias(t2,c);}
		   )
		 )
	;

rules
    :   ( rule )+
    ;

rule
    :   #( RULE id:ID {currentRuleName=#id.getText();}
           (m:modifier)?
           (ARG (ARG_ACTION)?)
           (RET (ARG_ACTION)?)
           (optionsSpec)?
           (ruleScopeSpec)?
       	   (AMPERSAND)*
           b:block
           (exceptionGroup)?
           EOR
           {trackTokenRule(#id,#m,#b);}
         )
    ;

modifier
	:	"protected"
	|	"public"
	|	"private"
	|	"fragment"
	;

ruleScopeSpec
 	:	#( "scope" (ACTION)? ( ID )* )
 	;

block
    :   #(  BLOCK
            (optionsSpec)?
            ( alternative rewrite )+
            EOB   
         )
    ;

alternative
    :   #( ALT (element)+ EOA )
    ;

exceptionGroup
	:	( exceptionHandler )+ (finallyClause)?
	|	finallyClause
    ;

exceptionHandler
    :    #("catch" ARG_ACTION ACTION)
    ;

finallyClause
    :    #("finally" ACTION)
    ;

rewrite
	:	( #( REWRITE (SEMPRED)? (ALT|TEMPLATE|ACTION|ETC) ) )*
	;

element
    :   #(ROOT element)
    |   #(BANG element)
    |   atom
    |   #(NOT element)
    |   #(RANGE atom atom)
    |   #(CHAR_RANGE atom atom)
    |	#(ASSIGN ID element)
    |	#(PLUS_ASSIGN ID element)
    |   ebnf
    |   tree
    |   #( SYNPRED block ) 
    |   FORCED_ACTION
    |   ACTION
    |   SEMPRED
    |   SYN_SEMPRED
    |   BACKTRACK_SEMPRED
    |   GATED_SEMPRED
    |   EPSILON 
    ;

ebnf:   block
    |   #( OPTIONAL block ) 
    |   #( CLOSURE block )  
    |   #( POSITIVE_CLOSURE block ) 
    ;

tree:   #(TREE_BEGIN  element (element)*  )
    ;

atom
    :   #( rr:RULE_REF (rarg:ARG_ACTION)? )
    |   #( t:TOKEN_REF (targ:ARG_ACTION )? ) {trackToken(t);}
    |   c:CHAR_LITERAL   {trackString(c);}
    |   s:STRING_LITERAL {trackString(s);}
    |   WILDCARD
    |   #(DOT ID atom) // scope override on rule
    ;

ast_suffix
	:	ROOT
	|	BANG
	;
