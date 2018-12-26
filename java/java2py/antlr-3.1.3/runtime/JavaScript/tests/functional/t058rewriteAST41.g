
grammar t058rewriteAST41;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : atom -> ^(atom atom) ; // NOT CYCLE! (dup atom)
atom : INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;