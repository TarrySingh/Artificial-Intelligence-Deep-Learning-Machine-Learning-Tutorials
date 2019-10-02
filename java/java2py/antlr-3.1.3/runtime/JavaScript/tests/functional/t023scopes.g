grammar t023scopes;

options {
    language=JavaScript;
}

prog
scope {
name
}
    :   ID {$prog::name=$ID.text;}
    ;

ID  :   ('a'..'z')+
    ;

WS  :   (' '|'\n'|'\r')+ {$channel=org.antlr.runtime.BaseRecognizer.HIDDEN;}
    ;
