grammar t056lexer1;
options {language=JavaScript;}
a : A {this.xlog(this.input);} ;
A : '\\' 't' {this.setText("  ");} ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
