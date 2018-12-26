grammar t056lexer4;
options {language=JavaScript;}
tokens {X;}
a : X EOF {this.xlog(this.input);} ;
A : '-' I {$type = this.X;} ;
I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
