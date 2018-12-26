
grammar t058rewriteAST72;
options {language=JavaScript;output=AST;}
a : ID+ -> ID ID ID ; // only 2 input IDs
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;