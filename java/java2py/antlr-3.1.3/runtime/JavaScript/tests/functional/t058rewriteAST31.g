
grammar t058rewriteAST31;
options {language=JavaScript;output=AST;}
tokens {VAR;}
a : ID (',' ID)*-> ^(VAR["var"] ID)+ ;
type : 'int' | 'float' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;