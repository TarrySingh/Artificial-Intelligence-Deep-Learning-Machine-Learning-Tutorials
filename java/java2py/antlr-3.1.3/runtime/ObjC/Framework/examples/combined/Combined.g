grammar Combined;

options {
	language=ObjC;
}

stat: identifier+  ;

identifier
    : ID
    ;


ID  :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
    ;

INT :   ('0'..'9')+
    ;

WS  :   (   ' '
        |   '\t'
        |   '\r'
        |   '\n'
        )+
        { $channel=99; }
    ;    
