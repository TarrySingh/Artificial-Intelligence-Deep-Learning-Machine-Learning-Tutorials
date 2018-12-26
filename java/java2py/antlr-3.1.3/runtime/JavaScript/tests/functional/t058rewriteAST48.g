
grammar t058rewriteAST48;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : x=b -> ^($x $x);
b : ID ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;