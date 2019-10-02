grammar t057autoAST16;
options {language=JavaScript;output=AST;}
a  : type ID ;
type : {pass}'int' | 'float' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
