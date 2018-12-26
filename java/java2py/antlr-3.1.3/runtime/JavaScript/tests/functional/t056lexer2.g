grammar t056lexer2;
options {language=JavaScript;}
a : A EOF {this.xlog(this.input);} ;
A : '-' I ;
I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
