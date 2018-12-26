grammar t056lexer7;
options {language=JavaScript;}
a : A EOF ;
A : 'hi' WS (v=I)? {$channel=0; this.xlog($v.text);} ;
fragment I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
