
grammar t058rewriteAST12;
options {language=JavaScript;output=AST;}
a : 'void' ID INT -> 'void' ^(INT ID);
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;