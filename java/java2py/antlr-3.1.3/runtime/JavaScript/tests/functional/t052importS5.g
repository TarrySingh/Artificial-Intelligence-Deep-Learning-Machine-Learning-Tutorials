parser grammar t052importS5;
options {
    language=JavaScript;
}
tokens { A; B; C; }
@members {
    this.capture = function(t) {
        this.gt052importM5.capture(t);
    };
}
x : A {this.capture("S.x ");} ;
