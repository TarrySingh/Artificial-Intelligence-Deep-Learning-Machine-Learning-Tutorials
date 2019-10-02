grammar t045dfabug;
options {
    language = JavaScript;
    output = AST;
}


r
options { backtrack=true; }
    : (modifier+ INT)=> modifier+ expression
    | modifier+ statement
    ;

expression
    : INT '+' INT
    ;

statement
    : 'fooze'
    | 'fooze2'
    ;

modifier
    : 'public'
    | 'private'
    ;

ID : 'a'..'z' + ;
INT : '0'..'9' +;
WS: (' ' | '\n' | '\t')+ {$channel = HIDDEN;};

