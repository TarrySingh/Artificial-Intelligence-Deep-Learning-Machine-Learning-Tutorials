grammar t014parser;
options {
  language = JavaScript;
}

@parser::members {
this.reportedErrors = [];
this.events = [];
this.emitErrorMessage = function(msg) {
    this.reportedErrors.push(msg);
};
this.eventMessage = function(msg) {
    this.events.push(msg);
};
}
        

document:
        ( declaration
        | call
        )*
        EOF
    ;

declaration:
        'var' t=IDENTIFIER ';'
        {this.eventMessage(['decl', $t.getText()]);}
    ;

call:
        t=IDENTIFIER '(' ')' ';'
        {this.eventMessage(['call', $t.getText()]);}
    ;

IDENTIFIER: ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*;
WS:  (' '|'\r'|'\t'|'\n') {$channel=HIDDEN;};
