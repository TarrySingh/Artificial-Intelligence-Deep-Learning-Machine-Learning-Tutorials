
grammar t058rewriteAST74;
options {language=JavaScript;output=AST;}
a : ID? INT -> ID+ INT ;
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;