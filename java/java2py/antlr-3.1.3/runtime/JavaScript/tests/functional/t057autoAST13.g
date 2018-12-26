grammar t057autoAST13;
options {language=JavaScript;output=AST;}
a : ID INT^ ID^ ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
