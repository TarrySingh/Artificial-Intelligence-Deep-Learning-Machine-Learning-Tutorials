grammar t057autoAST54;
options {language=JavaScript;output=AST;}
a : b | c ;
b : ID ;
c : INT ;
ID : 'a'..'z'+ ;
S : '*' ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
