
grammar t058rewriteAST75;
options {language=JavaScript;output=AST;}
a : ID c=. -> $c;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;