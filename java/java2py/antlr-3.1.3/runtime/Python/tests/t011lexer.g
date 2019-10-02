lexer grammar t011lexer;
options {
  language = Python;
}

IDENTIFIER: 
        ('a'..'z'|'A'..'Z'|'_') 
        ('a'..'z'
        |'A'..'Z'
        |'0'..'9'
        |'_'
            { 
              print "Underscore" 
              print "foo"
            }
        )*
    ;

WS: (' ' | '\n')+;
