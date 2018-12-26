
grammar t058rewriteAST27;
options {language=JavaScript;output=AST;}
a : op INT -> ^(INT op);
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;