
grammar t058rewriteAST5;
options {language=JavaScript;output=AST;}
a : ID -> ID[ ];
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;