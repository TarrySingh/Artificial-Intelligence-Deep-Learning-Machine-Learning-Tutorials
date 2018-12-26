grammar t021hoist;
options {
    language=Python;
}

/* With this true, enum is seen as a keyword.  False, it's an identifier */
@parser::init {
self.enableEnum = False
}

stat returns [enumIs]
    : identifier    {enumIs = "ID"}
    | enumAsKeyword {enumIs = "keyword"}
    ;

identifier
    : ID
    | enumAsID
    ;

enumAsKeyword : {self.enableEnum}? 'enum' ;

enumAsID : {not self.enableEnum}? 'enum' ;

ID  :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
    ;

INT :	('0'..'9')+
    ;

WS  :   (   ' '
        |   '\t'
        |   '\r'
        |   '\n'
        )+
        {$channel=HIDDEN}
    ;    
