parser grammar t052importS1;
options {
    language=JavaScript;
}
@members {
    this.capture = function(t) {
        this.gt052importM1.capture(t);
    };
}

a : B { this.capture("S.a") } ;
