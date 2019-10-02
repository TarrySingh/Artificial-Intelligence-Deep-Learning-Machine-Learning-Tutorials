grammar t057autoAST21;
options {language=JavaScript;output=AST;}
a : ID (op^ ID)* ;
op : {pass}'+' | '-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
