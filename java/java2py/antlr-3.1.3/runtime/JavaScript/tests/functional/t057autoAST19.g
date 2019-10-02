grammar t057autoAST19;
options {language=JavaScript;output=AST;}
a  : x+=type^ ID ;
type : {pass}'int' | 'float' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
