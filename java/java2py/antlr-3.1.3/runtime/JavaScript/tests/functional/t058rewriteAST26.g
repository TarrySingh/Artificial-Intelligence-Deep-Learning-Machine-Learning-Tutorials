
grammar t058rewriteAST26;
options {language=JavaScript;output=AST;}
a : op INT -> ^(op INT);
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;