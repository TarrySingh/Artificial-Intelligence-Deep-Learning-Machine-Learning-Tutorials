
grammar t058rewriteAST56;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : x=b -> $x;
b : ID ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;