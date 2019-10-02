
grammar t058rewriteAST69;
options {language=JavaScript;output=AST;}
tokens { FLOAT; }
r
    : INT -> {new org.antlr.runtime.tree.CommonTree(new org.antlr.runtime.CommonToken(FLOAT, $INT.text+".0"))} 
    ; 
INT : '0'..'9'+; 
WS: (' ' | '\n' | '\t')+ {$channel = HIDDEN;}; 