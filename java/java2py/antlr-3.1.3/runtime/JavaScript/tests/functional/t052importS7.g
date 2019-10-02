lexer grammar t052importS7;
options {
    language=JavaScript;
}
@members {
    this.capture = function(t) {
        this.gt052importM7.capture(t);
    };
}
A : 'a' {this.capture("S.A ");} ;
C : 'c' ;
