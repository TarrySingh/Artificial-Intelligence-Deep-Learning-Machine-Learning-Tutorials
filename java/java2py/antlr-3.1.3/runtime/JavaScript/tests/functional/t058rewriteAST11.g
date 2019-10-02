
grammar t058rewriteAST11;
options {language=JavaScript;output=AST;}
a : ID INT -> ^(INT ID);
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;