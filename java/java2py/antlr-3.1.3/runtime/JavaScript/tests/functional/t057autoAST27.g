grammar t057autoAST27;
options {language=JavaScript;output=AST;}
a : ~ID '+' INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
