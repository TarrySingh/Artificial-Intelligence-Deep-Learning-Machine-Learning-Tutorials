lexer grammar t052importS8;
options {
    language=JavaScript;
}
@members {
    this.capture = function(t) {
        this.gt052importM8.capture(t);
    };
}
A : 'a' {this.capture("S.A");} ;
