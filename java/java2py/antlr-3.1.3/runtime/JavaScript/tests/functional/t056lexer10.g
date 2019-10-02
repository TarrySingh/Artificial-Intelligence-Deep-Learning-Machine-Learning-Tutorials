grammar t056lexer10;
options {language=JavaScript;}
a : A ;
A : i+=I WS i+=I {$channel=0; for (var p=0; p<$i.length; p++) this.xlog(" "+$i[p].getText()); } ;
fragment I : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
