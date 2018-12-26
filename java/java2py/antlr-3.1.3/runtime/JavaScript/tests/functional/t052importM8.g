lexer grammar t052importM8;
options {
    language=JavaScript;
}
import t052importS8;
A : 'a' {this.capture("M.A ");} ;
WS : (' '|'\n') {this.skip();} ;
