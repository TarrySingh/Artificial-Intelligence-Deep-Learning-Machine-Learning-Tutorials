grammar t057autoAST43;
options {language=JavaScript;output=AST;}
a : A b=B b=B c+=C c+=C D {s = $D.text} ;
A : 'a' ;
B : 'b' ;
C : 'c' ;
D : 'd' ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
