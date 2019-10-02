grammar t044trace;
options {
  language = Python;
}

@init {
    self._stack = None
}

a: '<' ((INT '+')=>b|c) '>';
b: c ('+' c)*;
c: INT 
    {
        if self._stack is None:
            self._stack = self.getRuleInvocationStack()
    }
    ;

INT: ('0'..'9')+;
WS: (' ' | '\n' | '\t')+ {$channel = HIDDEN;};
