
grammar t058rewriteAST45;
options {language=JavaScript;output=AST;}
tokens {MOD;}
a : modifier? type ID (',' ID)* ';' -> ^(type ^(MOD modifier)? ID)+ ;
type : 'int' ;
modifier : 'public' ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;