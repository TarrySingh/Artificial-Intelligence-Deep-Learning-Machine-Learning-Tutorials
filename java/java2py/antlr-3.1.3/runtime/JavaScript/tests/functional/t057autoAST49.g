grammar t057autoAST49;
options {language=JavaScript;output=AST;}
a : ID INT ; // follow is EOF
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
