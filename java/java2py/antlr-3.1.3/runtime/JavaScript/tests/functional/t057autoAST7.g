grammar t057autoAST7;
options {language=JavaScript;output=AST;}
a : v='void'^ ID ';' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
