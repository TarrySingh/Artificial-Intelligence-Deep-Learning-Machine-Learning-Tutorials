
grammar t058rewriteAST83;
options {language=JavaScript;output=AST;}
a : b -> b | c -> c;
b : ID -> ID ;
c : INT -> INT ;
ID : 'a'..'z'+ ;
S : '*' ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;