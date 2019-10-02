grammar t057autoAST11;
options {language=JavaScript;output=AST;}
a : v='void' x=.^ ';' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
