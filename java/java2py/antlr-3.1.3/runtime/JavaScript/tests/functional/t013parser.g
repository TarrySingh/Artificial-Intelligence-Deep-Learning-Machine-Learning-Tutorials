grammar t013parser;
options {
  language = JavaScript;
}

@parser::members {
this.identifiers = [];
this.reportedErrors = [];

this.foundIdentifier = function(name) {
    this.identifiers.push(name);
};

this.emitErrorMessage = function(msg) {
    this.reportedErrors.push(msg);
};
}

document:
        t=IDENTIFIER {this.foundIdentifier($t.text)}
        ;

IDENTIFIER: ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*;
