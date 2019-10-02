grammar t057autoAST8;
options {language=JavaScript;output=AST;}
a : v='void'^ . ';' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
