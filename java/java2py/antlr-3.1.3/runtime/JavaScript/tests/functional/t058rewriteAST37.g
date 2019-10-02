
grammar t058rewriteAST37;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : b b ;
b : ID ( ID (last=ID -> $last)+ ) ';' // get last ID
  | INT // should still get auto AST construction
  ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;