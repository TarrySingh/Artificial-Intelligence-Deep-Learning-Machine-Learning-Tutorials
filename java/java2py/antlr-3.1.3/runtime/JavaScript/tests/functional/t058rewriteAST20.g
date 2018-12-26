
grammar t058rewriteAST20;
options {language=JavaScript;output=AST;}
a : (x=ID)? -> $x?;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;