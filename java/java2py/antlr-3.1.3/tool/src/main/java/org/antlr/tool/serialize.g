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
	package org.antlr.tool;
	import java.util.*;
	import org.antlr.analysis.*;
	import org.antlr.misc.*;
	import java.io.*;
}

class SerializerWalker extends TreeParser;

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
            "serialize: "+ex.toString(),
            ex);
    }

protected Grammar grammar;
protected String currentRuleName;
protected GrammarSerializer out;
}

grammar[GrammarSerializer out]
{
	this.out = out;
}
    :   #( LEXER_GRAMMAR 	grammarSpec[#grammar.getType()] )
	|   #( PARSER_GRAMMAR   grammarSpec[#grammar.getType()] )
	|   #( TREE_GRAMMAR     grammarSpec[#grammar.getType()] )
	|   #( COMBINED_GRAMMAR grammarSpec[#grammar.getType()] )
    ;

grammarSpec[int gtokentype]
	:	id:ID {out.grammar(gtokentype, #id.getText());}
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

optionsSpec
    :   #( OPTIONS (option)+ )
    ;

option
    :   #( ASSIGN ID optionValue )
    ;

optionValue 
    :   id:ID
    |   s:STRING_LITERAL
    |   c:CHAR_LITERAL
    |   i:INT
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
	:	t:TOKEN_REF
	|	#( ASSIGN
		   t2:TOKEN_REF      
		   ( s:STRING_LITERAL
		   | c:CHAR_LITERAL 
		   )
		 )
	;

rules
    :   ( rule )+
    ;

rule
    :   #( RULE id:ID           {out.rule(#id.getText());}
           (m:modifier)?
           (ARG (ARG_ACTION)?)
           (RET (ARG_ACTION)?)
           (optionsSpec)?
           (ruleScopeSpec)?
       	   (AMPERSAND)*
           b:block
           (exceptionGroup)?
           EOR                  {out.endRule();}
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
    :   #(  BLOCK {out.block(#BLOCK.getNumberOfChildrenWithType(ALT));}
            (optionsSpec)?
            ( alternative rewrite )+
            EOB   
         )
    ;

alternative
    :   #( ALT {out.alt(#alternative);} (element)+ EOA {out.endAlt();} )
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
    |   #(NOT {out.not();} element)
    |   #(RANGE atom atom)
    |   #(CHAR_RANGE {out.range();} atom atom)
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
    :   #( rr:RULE_REF (rarg:ARG_ACTION)? )     {out.ruleRef(#rr);}
    |   #( t:TOKEN_REF (targ:ARG_ACTION )? )    {out.token(#t);}
    |   c:CHAR_LITERAL                          {out.charLiteral(#c);}
    |   s:STRING_LITERAL                        {out.charLiteral(#s);}
    |   WILDCARD                                {out.wildcard(#WILDCARD);}
    |   #(DOT ID atom) // scope override on rule
    ;

ast_suffix
	:	ROOT
	|	BANG
	;
