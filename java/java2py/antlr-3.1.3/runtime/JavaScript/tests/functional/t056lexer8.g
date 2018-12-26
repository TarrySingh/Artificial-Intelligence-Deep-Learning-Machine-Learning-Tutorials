grammar t056lexer8;
options {language=JavaScript;}
a : A EOF ;
A : I {this.xlog($I.text);} ;
fragment I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
