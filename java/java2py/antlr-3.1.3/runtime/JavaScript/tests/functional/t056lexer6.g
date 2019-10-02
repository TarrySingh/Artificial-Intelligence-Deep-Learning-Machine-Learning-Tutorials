grammar t056lexer6;
options {language=JavaScript;}
a : A EOF {this.xlog(this.input);} ;
A : I '.' I ;
fragment I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
