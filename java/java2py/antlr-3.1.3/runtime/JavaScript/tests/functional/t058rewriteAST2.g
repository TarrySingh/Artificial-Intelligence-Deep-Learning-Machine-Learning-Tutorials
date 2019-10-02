
grammar t058rewriteAST2;
options {language=JavaScript;output=AST;}
a : ID -> ID;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;