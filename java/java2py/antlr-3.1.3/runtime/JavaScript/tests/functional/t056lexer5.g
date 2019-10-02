grammar t056lexer5;
options {language=JavaScript;}
a : A {this.xlog(this.input);} ;
A : '-' I ;
fragment I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
