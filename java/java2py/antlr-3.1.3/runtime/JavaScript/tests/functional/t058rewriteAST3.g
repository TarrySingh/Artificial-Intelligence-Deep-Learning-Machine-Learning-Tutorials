
grammar t058rewriteAST3;
options {language=JavaScript;output=AST;}
a : ID -> ID["x"];
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;