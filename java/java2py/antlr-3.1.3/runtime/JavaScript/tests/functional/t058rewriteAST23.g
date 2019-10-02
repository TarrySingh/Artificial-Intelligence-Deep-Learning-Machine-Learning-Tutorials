
grammar t058rewriteAST23;
options {language=JavaScript;output=AST;}
a : ID -> {false}? ID -> ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;