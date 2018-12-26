grammar t057autoAST15;
options {language=JavaScript;output=AST;}
a : 'void' (({pass}ID|INT) ID | 'null' ) ';' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
