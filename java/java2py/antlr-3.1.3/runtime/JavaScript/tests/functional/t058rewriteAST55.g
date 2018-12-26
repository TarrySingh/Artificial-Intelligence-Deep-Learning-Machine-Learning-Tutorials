
grammar t058rewriteAST55;
options {language=JavaScript;output=AST;}
tokens {BLOCK;}
a : A A b=B B b=B c+=C C c+=C D {s=$D.text} -> A+ B+ C+ D ;
type : 'int' | 'float' ;
A : 'a' ;
B : 'b' ;
C : 'c' ;
D : 'd' ;
WS : (' '|'\n') {$channel=HIDDEN;} ;