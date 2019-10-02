
grammar t058rewriteAST82;
options {language=JavaScript;output=AST;}
a : b c -> b c;
b : ID -> ID ;
c : INT -> INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;