
grammar t058rewriteAST8;
options {language=JavaScript;output=AST;}
a : b -> b;
b : ID ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;