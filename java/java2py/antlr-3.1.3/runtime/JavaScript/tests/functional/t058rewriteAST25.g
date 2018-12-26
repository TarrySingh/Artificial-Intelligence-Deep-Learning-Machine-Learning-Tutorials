
grammar t058rewriteAST25;
options {language=JavaScript;output=AST;}
a : ID INT -> {false}? ^(ID INT)
           -> {true}? ^(INT ID)
           -> ID
  ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;