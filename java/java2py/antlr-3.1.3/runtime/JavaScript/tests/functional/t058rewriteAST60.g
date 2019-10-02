
grammar t058rewriteAST60;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : x=b (y=b)? -> $x $y?;
b : ID ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;