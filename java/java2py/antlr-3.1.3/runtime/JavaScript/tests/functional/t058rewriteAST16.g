
grammar t058rewriteAST16;
options {language=JavaScript;output=AST;}
a : ID ID -> ID+ ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;