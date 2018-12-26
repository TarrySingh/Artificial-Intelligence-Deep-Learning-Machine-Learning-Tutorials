grammar t057autoAST38;
options {language=JavaScript;output=AST;}
a : id+=ID^ ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
