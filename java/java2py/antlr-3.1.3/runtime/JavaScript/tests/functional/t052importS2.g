parser grammar t052importS2;
options {
    language=JavaScript;
}
@members {
    this.capture = function(t) {
        this.gt052importM2.capture(t);
    }
}
a[x] returns [y] : B {this.capture("S.a"); $y="1000";} ;
