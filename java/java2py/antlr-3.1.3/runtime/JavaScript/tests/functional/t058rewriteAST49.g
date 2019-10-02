
grammar t058rewriteAST49;
options {language=JavaScript;output=AST;}
a : 'int' ID (',' ID)* ';' -> ^('int' ID+) ;
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;