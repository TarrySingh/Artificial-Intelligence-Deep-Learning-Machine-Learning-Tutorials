
grammar t058rewriteAST73;
options {language=JavaScript;output=AST;}
a : ID? INT -> ID INT ;
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;