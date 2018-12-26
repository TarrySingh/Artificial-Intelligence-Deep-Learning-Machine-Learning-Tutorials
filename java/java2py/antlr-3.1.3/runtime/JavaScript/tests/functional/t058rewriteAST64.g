
grammar t058rewriteAST64;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : ID -> ID? ; // match an ID to optional ID
b : ID ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;