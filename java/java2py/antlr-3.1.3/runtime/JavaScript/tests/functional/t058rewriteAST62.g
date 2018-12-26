
grammar t058rewriteAST62;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : x=ID (y=b)? -> ($x $y)?;
b : ID ;
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;