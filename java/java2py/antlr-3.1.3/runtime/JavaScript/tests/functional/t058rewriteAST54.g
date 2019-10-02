
grammar t058rewriteAST54;
options {language=JavaScript;output=AST;}
tokens {VAR;}
a : first=ID others+=ID* -> $first VAR $others+ ;
op : '+'|'-' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;