parser grammar t052importT5;
options {
    language=JavaScript;
}
tokens { C; B; A; } /// reverse order
@members {
    this.capture = function(t) {
        this.gt052importM5.capture(t);
    };
}
y : A {this.capture("T.y");} ;
