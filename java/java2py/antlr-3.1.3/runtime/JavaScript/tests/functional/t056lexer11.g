grammar t056lexer11;
options {language=JavaScript;}
a : B ;
B : x='a' {this.xlog($x);} ;
