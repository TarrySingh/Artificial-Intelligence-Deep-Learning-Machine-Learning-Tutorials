
grammar t058rewriteAST35;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : b b ;
b : (ID INT -> INT ID | INT INT -> INT+ )
  ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;