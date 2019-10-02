grammar t030specialStates;
options {
  language = JavaScript;
}

@members {
this.recover = function(input, re) {
    throw re;
};
}

r
    : ( {this.cond}? NAME
        | {!this.cond}? NAME WS+ NAME
        )
        ( WS+ NAME )?
        EOF
    ;

NAME: ('a'..'z') ('a'..'z' | '0'..'9')+;
NUMBER: ('0'..'9')+;
WS: ' '+;
