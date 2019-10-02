parser grammar t052importS6;
options {
    language=JavaScript;
}
@members {
    this.capture = function(t) {
        this.gt052importM6.capture(t);
    };
}
a : b { this.capture("S.a") } ;
b : B ;
