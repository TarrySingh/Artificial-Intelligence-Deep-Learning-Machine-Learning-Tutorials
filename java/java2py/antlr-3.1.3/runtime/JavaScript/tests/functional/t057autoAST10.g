grammar t057autoAST10;
options {language=JavaScript;output=AST;}
a : v='void' x=.^ ';' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
