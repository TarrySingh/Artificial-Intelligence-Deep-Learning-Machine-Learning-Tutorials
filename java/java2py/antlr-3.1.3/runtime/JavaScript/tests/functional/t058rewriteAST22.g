
grammar t058rewriteAST22;
options {language=JavaScript;output=AST;}
a : ID -> {true}? ID -> ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;