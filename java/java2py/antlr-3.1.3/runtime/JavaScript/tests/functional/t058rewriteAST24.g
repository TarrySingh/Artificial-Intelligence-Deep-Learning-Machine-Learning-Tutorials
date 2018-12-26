
grammar t058rewriteAST24;
options {language=JavaScript;output=AST;}
a : ID INT -> {false}? ID
           -> {true}? INT
           -> 
  ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;