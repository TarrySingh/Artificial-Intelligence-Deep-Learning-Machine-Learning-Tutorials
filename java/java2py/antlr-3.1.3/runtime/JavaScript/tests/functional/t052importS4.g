parser grammar t052importS4;
options {
    language=JavaScript;
}
@members {
    this.capture = function(t) {
        this.gt052importM4.capture(t);
    };
}
a : b {this.capture("S.a");} ;
b : B ;
