lexer grammar t025lexerRulePropertyRef;
options {
  language = JavaScript;
}

@lexer::init {
this.properties = [];
}

IDENTIFIER: 
        ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
        {
this.properties.push(
    [$text, $type, $line, $pos, $index, $channel, $start, $stop]
);
        }
    ;
WS: (' ' | '\n')+;
