
grammar t058rewriteAST10;
options {language=JavaScript;output=AST;}
a : b INT -> INT b;
b : ID ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;