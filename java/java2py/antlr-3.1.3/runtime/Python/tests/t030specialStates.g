grammar t030specialStates;
options {
  language = Python;
}

@init {
self.cond = True
}

@members {
def recover(self, input, re):
    # no error recovery yet, just crash!
    raise re
}

r
    : ( {self.cond}? NAME
        | {not self.cond}? NAME WS+ NAME
        )
        ( WS+ NAME )?
        EOF
    ;

NAME: ('a'..'z') ('a'..'z' | '0'..'9')+;
NUMBER: ('0'..'9')+;
WS: ' '+;
