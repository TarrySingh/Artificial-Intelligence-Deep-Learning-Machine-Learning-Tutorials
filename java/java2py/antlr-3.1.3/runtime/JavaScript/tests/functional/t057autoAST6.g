grammar t057autoAST6;
options {language=JavaScript;output=AST;}
a : ( ID INT )? ID^ ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
