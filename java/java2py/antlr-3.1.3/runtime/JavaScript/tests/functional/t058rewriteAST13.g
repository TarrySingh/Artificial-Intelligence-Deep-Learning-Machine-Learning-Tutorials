
grammar t058rewriteAST13;
options {language=JavaScript;output=AST;}
tokens {DUH;}
a : ID INT ID INT -> ^( DUH ID ^( DUH INT) )+ ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;