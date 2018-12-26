grammar t057autoAST25;
options {language=JavaScript;output=AST;}
a : x=('+' | '-')^ ID ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
