
grammar t058rewriteAST6;
options {language=JavaScript;output=AST;}
a : 'c' -> 'c';
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;