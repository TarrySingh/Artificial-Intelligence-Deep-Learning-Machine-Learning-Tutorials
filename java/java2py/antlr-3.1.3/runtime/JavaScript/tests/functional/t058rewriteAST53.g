
grammar t058rewriteAST53;
options {language=JavaScript;output=AST;}
a : ID+ -> ID ID ID ; // works if 3 input IDs
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;