
grammar t058rewriteAST7;
options {language=JavaScript;output=AST;}
a : 'ick' -> 'ick';
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;