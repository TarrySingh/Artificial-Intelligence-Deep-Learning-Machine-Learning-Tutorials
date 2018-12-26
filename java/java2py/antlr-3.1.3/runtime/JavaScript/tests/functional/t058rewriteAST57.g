
grammar t058rewriteAST57;
options {language=JavaScript;output=AST;}
a : ID a -> a | INT ;
ID : 'a'..'z'+ ;
INT: '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;