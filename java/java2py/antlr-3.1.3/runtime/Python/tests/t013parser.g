grammar t013parser;
options {
  language = Python;
}

@parser::init {
self.identifiers = []
self.reportedErrors = []
}

@parser::members {
def foundIdentifier(self, name):
    self.identifiers.append(name)

def emitErrorMessage(self, msg):
    self.reportedErrors.append(msg)
}

document:
        t=IDENTIFIER {self.foundIdentifier($t.text)}
        ;

IDENTIFIER: ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*;
