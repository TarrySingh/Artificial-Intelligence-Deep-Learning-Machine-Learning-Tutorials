grammar t057autoAST50;
options {language=JavaScript;output=AST;}
a : b ;
b : ID INT ; // follow should see EOF
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
