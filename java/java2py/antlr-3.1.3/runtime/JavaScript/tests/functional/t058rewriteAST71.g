
grammar t058rewriteAST71;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : ID ID INT INT INT -> (ID INT)+;
ID : 'a'..'z'+ ;
INT : '0'..'9'+; 
WS : (' '|'\n') {$channel=HIDDEN;} ;