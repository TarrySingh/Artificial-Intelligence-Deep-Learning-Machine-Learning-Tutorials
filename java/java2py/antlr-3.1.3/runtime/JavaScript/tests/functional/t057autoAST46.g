grammar t057autoAST46;
options {language=JavaScript;output=AST;}
decl : type^ ID '='! INT ';'! ;
type : 'int' | 'float' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
