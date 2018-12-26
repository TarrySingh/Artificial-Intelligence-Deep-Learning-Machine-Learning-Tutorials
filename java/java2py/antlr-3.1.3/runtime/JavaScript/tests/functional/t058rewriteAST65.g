
grammar t058rewriteAST65;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : x+=b x+=b -> {new org.antlr.runtime.tree.CommonTree(null)};
b : ID ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;