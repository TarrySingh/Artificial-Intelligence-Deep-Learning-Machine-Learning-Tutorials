
grammar t058rewriteAST43;
options {language=JavaScript;output=AST;}
a : modifier? type ID (',' ID)* ';' -> ^(type modifier? ID)+ ;
type : 'int' ;
modifier : 'public' ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;