
grammar t058rewriteAST4;
options {language=JavaScript;output=AST;}
a : ID -> ^(ID["x"] INT);
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;