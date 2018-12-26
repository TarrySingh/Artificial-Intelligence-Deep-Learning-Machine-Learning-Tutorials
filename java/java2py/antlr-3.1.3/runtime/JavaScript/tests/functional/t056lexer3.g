grammar t056lexer3;
options {language=JavaScript;}
a : A EOF {this.xlog($A.text+", channel="+$A.channel);} ;
A : '-' WS I ;
I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
