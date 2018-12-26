
grammar t058rewriteAST46;
options {language=JavaScript;output=AST;}
tokens {MOD;}
a : ID (',' ID)* ';' -> ID+ ID+ ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;