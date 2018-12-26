
grammar t058rewriteAST32;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : lc='{' ID+ '}' -> ^(BLOCK[$lc] ID+) ;
type : 'int' | 'float' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;