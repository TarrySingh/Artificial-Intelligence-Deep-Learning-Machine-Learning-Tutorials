
grammar t058rewriteAST34;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : b b^ ; // 2nd b matches only an INT; can make it root
b : ID INT -> INT ID
  | INT
  ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;