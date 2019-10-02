grammar t057autoAST51;
options {language=JavaScript;output=AST;}
a : b c ;
b : ID ;
c : INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
