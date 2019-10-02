grammar t057autoAST31;
options {language=JavaScript;output=AST;}
a : ~'+'^ INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
