
grammar t058rewriteAST19;
options {language=JavaScript;output=AST;}
a : x+=b x+=b -> $x*;
b : ID ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;