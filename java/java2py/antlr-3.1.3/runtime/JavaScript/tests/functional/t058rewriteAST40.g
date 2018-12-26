
grammar t058rewriteAST40;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : (atom -> atom) (op='+' r=atom -> ^($op $a $r) )* ;
atom : INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;