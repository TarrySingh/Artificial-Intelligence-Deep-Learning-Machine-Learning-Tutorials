/*
 [The "BSD licence"]
 Copyright (c) 2005-2007 Terence Parr
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

/** ANTLR v3 tree grammar to walk trees created by ANTLRv3.g */
tree grammar ANTLRv3Tree;

options {
	tokenVocab = ANTLRv3;
	ASTLabelType = CommonTree;
}

@header {
package org.antlr.grammar.v3;
}

grammarDef
    :   ^( grammarType ID DOC_COMMENT? optionsSpec? tokensSpec? attrScope* action* rule+ )
    ;

grammarType
	:	LEXER_GRAMMAR
    |	PARSER_GRAMMAR
    |	TREE_GRAMMAR
    |	COMBINED_GRAMMAR
    ;

tokensSpec
	:	^(TOKENS tokenSpec+)
	;

tokenSpec
	:	^('=' TOKEN_REF STRING_LITERAL)
	|	^('=' TOKEN_REF CHAR_LITERAL)
	|	TOKEN_REF
	;

attrScope
	:	^('scope' ID ACTION)
	;

action
	:	^('@' ID ID ACTION)
	|	^('@' ID ACTION)
	;

optionsSpec
	:	^(OPTIONS option+)
	;

option
    :   qid // only allowed in element options
    |	^('=' ID optionValue)
 	;
 	
optionValue
    :   ID
    |   STRING_LITERAL
    |   CHAR_LITERAL
    |   INT
    ;

rule
	:	^( RULE ID modifier? (^(ARG ARG_ACTION))? (^(RET ARG_ACTION))?
	       throwsSpec? optionsSpec? ruleScopeSpec? ruleAction*
	       altList
	       exceptionGroup? EOR
	     )
	;

modifier
	:	'protected'|'public'|'private'|'fragment'
	;

/** Match stuff like @init {int i;} */
ruleAction
	:	^('@' ID ACTION)
	;

throwsSpec
	:	^('throws' ID+)
	;

ruleScopeSpec
	:	^('scope' ACTION)
	|	^('scope' ACTION ID+)
	|	^('scope' ID+)
	;

block
    :   ^( BLOCK optionsSpec? (alternative rewrite)+ EOB )
    ;

altList
    :   ^( BLOCK (alternative rewrite)+ EOB )
    ;

alternative
    :   ^(ALT element+ EOA)
    |   ^(ALT EPSILON EOA)
    ;

exceptionGroup
	:	exceptionHandler+ finallyClause?
	|	finallyClause
    ;

exceptionHandler
    :    ^('catch' ARG_ACTION ACTION)
    ;

finallyClause
    :    ^('finally' ACTION)
    ;

element
	:	^(('='|'+=') ID block)
	|	^(('='|'+=') ID atom)
	|	atom
	|	ebnf
	|   ACTION
	|   SEMPRED
	|	GATED_SEMPRED
	|   ^(TREE_BEGIN element+)
	;

atom:   ^(('^'|'!') atom)
	|	^(CHAR_RANGE CHAR_LITERAL CHAR_LITERAL optionsSpec?)
	|	^('~' notTerminal optionsSpec?)
	|	^('~' block optionsSpec?)
    |	^(RULE_REF ARG_ACTION)
    |	RULE_REF
    |   CHAR_LITERAL
    |   ^(CHAR_LITERAL optionsSpec)
    |	TOKEN_REF
    |	^(TOKEN_REF optionsSpec)
    |	^(TOKEN_REF ARG_ACTION optionsSpec)
    |	^(TOKEN_REF ARG_ACTION)
    |	STRING_LITERAL
    |	^(STRING_LITERAL optionsSpec)
    |	'.'
    |	^('.' optionsSpec?)
    ;

/** Matches ENBF blocks (and token sets via block rule) */
ebnf
	:	^(SYNPRED block)
	|	^(OPTIONAL block)
  	|	^(CLOSURE block)
   	|	^(POSITIVE_CLOSURE block)
	|	SYN_SEMPRED
	|	block
	;

notTerminal
	:   CHAR_LITERAL
	|	TOKEN_REF
	|	STRING_LITERAL
	;
		
// R E W R I T E  S Y N T A X

rewrite
	:	(^('->' SEMPRED rewrite_alternative))* ^('->' rewrite_alternative)
	|
	;

rewrite_alternative
	:	rewrite_template
	|	rewrite_tree_alternative
   	|   ^(ALT EPSILON EOA)
	;
	
rewrite_tree_block
    :   ^(BLOCK rewrite_tree_alternative EOB)
    ;

rewrite_tree_alternative
    :	^(ALT rewrite_tree_element+ EOA)
    ;

rewrite_tree_element
	:	rewrite_tree_atom
	|	rewrite_tree
	|   rewrite_tree_block
	|   rewrite_tree_ebnf
	;

rewrite_tree_atom
    :   CHAR_LITERAL
	|   TOKEN_REF
	|   ^(TOKEN_REF ARG_ACTION) // for imaginary nodes
    |   RULE_REF
	|   STRING_LITERAL
	|   LABEL
	|	ACTION
	;

rewrite_tree_ebnf
	:	^(OPTIONAL rewrite_tree_block)
  	|	^(CLOSURE rewrite_tree_block)
   	|	^(POSITIVE_CLOSURE rewrite_tree_block)
	;
	
rewrite_tree
	:	^(TREE_BEGIN rewrite_tree_atom rewrite_tree_element* )
	;

rewrite_template
	:   ^( TEMPLATE ID rewrite_template_args
		   (DOUBLE_QUOTE_STRING_LITERAL | DOUBLE_ANGLE_STRING_LITERAL)
		 )
	|	rewrite_template_ref
	|	rewrite_indirect_template_head
	|	ACTION
	;

/** foo(a={...}, ...) */
rewrite_template_ref
	:	^(TEMPLATE ID rewrite_template_args)
	;

/** ({expr})(a={...}, ...) */
rewrite_indirect_template_head
	:	^(TEMPLATE ACTION rewrite_template_args)
	;

rewrite_template_args
	:	^(ARGLIST rewrite_template_arg+)
	|	ARGLIST
	;

rewrite_template_arg
	:   ^(ARG ID ACTION)
	;

qid	:	ID ('.' ID)* ;
