grammar t026actions;
options {
  language = Python;
}

@lexer::init {
    self.foobar = 'attribute;'
}

prog
@init {
    self.capture('init;')
}
@after {
    self.capture('after;')
}
    :   IDENTIFIER EOF
    ;
    catch [ RecognitionException, exc ] {
        self.capture('catch;')
        raise
    }
    finally {
        self.capture('finally;')
    }


IDENTIFIER
    : ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
        {
            # a comment
          self.capture('action;')
            self.capture('\%r \%r \%r \%r \%r \%r \%r \%r;' \% ($text, $type, $line, $pos, $index, $channel, $start, $stop))
            if True:
                self.capture(self.foobar)
        }
    ;

WS: (' ' | '\n')+;
