grammar t057autoAST22;
options {language=JavaScript;output=AST;}
s : a ;
a : atom ('exp'^ a)? ;
atom : INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
