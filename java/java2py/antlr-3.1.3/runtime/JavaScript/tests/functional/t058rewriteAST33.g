
grammar t058rewriteAST33;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : lc='{' ID+ '}' -> ^(BLOCK[$lc,"block"] ID+) ;
type : 'int' | 'float' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;