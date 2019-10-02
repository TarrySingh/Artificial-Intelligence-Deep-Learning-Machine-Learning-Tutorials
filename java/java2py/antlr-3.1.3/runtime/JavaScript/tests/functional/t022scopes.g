grammar t022scopes;

options {
    language=JavaScript;
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
    $c::symbols = {};
}
    : '{' c1* c2+ '}'
        { $res = $c::symbols; }
    ;

c1
    : 'int' ID {$c::symbols[$ID.text] = true;} ';'
    ;

c2
    : ID '=' NUM ';'
        {
            if (! $c::symbols[$ID.text]) {
                throw new Error($ID.text);
            }
        }
    ;

/* recursive rule scopes, from the book, final beta, p.150 */

d returns [res]
scope {
    symbols
}
@init {
    $d::symbols = {};
}
    : '{' d1* d2* '}'
        { $res = $d::symbols; }
    ;

d1
    : 'int' ID {$d::symbols[$ID.text] = true;} ';'
    ;

d2
    : ID '=' NUM ';'
        {
            var i, isDefined;
            for (i=$d.length-1, isDefined=false; i>=0; i--) {
                if ($d[i]::symbols[$ID.text]) {
                    isDefined = true;
                    break;
                }
            }
            if (!isDefined) {
                throw new Error("undefined variable "+$ID.text);
            }
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
    : NUM { $e[0]::a = parseInt($NUM.text, 10); }
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
    : NUM { var len = $f.length-2; $f[len>=0 ? len : 0]::a = parseInt($NUM.text, 10); }
    | '{' f '}'
    ;


/* tokens */
ID  :   ('a'..'z')+
    ;

NUM :   ('0'..'9')+
    ;

WS  :   (' '|'\n'|'\r')+ {$channel=HIDDEN;}
    ;
