
grammar t058rewriteAST30;
options {language=JavaScript;output=AST;}
tokens {VAR;}
a : b -> ID ;
b : ID ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;