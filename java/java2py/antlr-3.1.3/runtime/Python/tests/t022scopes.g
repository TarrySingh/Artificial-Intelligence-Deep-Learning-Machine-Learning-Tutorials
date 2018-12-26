grammar t022scopes;

options {
    language=Python;
}

/* global scopes */

scope aScope {
names
}

a
scope aScope;
    :   {$aScope::names = [];} ID*
    ;


/* rule scopes, from the book, final beta, p.147 */

b[v]
scope {x}
    : {$b::x = v;} b2
    ;

b2
    : b3
    ;

b3 
    : {$b::x}?=> ID // only visible, if b was called with True
    | NUM
    ;


/* rule scopes, from the book, final beta, p.148 */

c returns [res]
scope {
    symbols
}
@init {
    $c::symbols = set();
}
    : '{' c1* c2+ '}'
        { $res = $c::symbols; }
    ;

c1
    : 'int' ID {$c::symbols.add($ID.text)} ';'
    ;

c2
    : ID '=' NUM ';'
        {
            if $ID.text not in $c::symbols:
                raise RuntimeError($ID.text)
        }
    ;

/* recursive rule scopes, from the book, final beta, p.150 */

d returns [res]
scope {
    symbols
}
@init {
    $d::symbols = set();
}
    : '{' d1* d2* '}'
        { $res = $d::symbols; }
    ;

d1
    : 'int' ID {$d::symbols.add($ID.text)} ';'
    ;

d2
    : ID '=' NUM ';'
        {
            for s in reversed(range(len($d))):
                if $ID.text in $d[s]::symbols:
                    break
            else:
                raise RuntimeError($ID.text)
        }
    | d
    ;

/* recursive rule scopes, access bottom-most scope */

e returns [res]
scope {
    a
}
@after {
    $res = $e::a;
}
    : NUM { $e[0]::a = int($NUM.text); }
    | '{' e '}'
    ;


/* recursive rule scopes, access with negative index */

f returns [res]
scope {
    a
}
@after {
    $res = $f::a;
}
    : NUM { $f[-2]::a = int($NUM.text); }
    | '{' f '}'
    ;


/* tokens */

ID  :   ('a'..'z')+
    ;

NUM :   ('0'..'9')+
    ;

WS  :   (' '|'\n'|'\r')+ {$channel=HIDDEN}
    ;
