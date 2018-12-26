lexer grammar t011lexer;
options {
  language = JavaScript;
}

IDENTIFIER: 
        ('a'..'z'|'A'..'Z'|'_') 
        ('a'..'z'
        |'A'..'Z'
        |'0'..'9'
        |'_'
            { 
              tlog("Underscore");
              tlog("foo");
            }
        )*
    ;

WS: (' ' | '\n')+;
