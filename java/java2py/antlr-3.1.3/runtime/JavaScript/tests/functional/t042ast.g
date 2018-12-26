grammar t042ast;
options {
    language = JavaScript;
    output = AST;
}

tokens {
    VARDEF;
    FLOAT;
    EXPR;
    BLOCK;
    VARIABLE;
    FIELD;
    CALL;
    INDEX;
    FIELDACCESS;
}

@header {
org.antlr.lang.map = function(a, fn) {
    var i, len, r=[];
    for (i=0, len=a.length; i<len; i++) {
        r.push(fn(a[i], i));
    }
    return r;
};
}

r1
    : INT ('+'^ INT)*
    ;

r2
    : 'assert'^ x=expression (':'! y=expression)? ';'!
    ;

r3
    : 'if'^ expression s1=statement ('else'! s2=statement)?
    ;

r4
    : 'while'^ expression statement
    ;

r5
    : 'return'^ expression? ';'!
    ;

r6
    : (INT|ID)+
    ;

r7
    : INT -> 
    ;

r8
    : 'var' ID ':' type -> ^('var' type ID) 
    ;

r9
    : type ID ';' -> ^(VARDEF type ID) 
    ;

r10
    : INT -> {new org.antlr.runtime.tree.CommonTree(new org.antlr.runtime.CommonToken(FLOAT, $INT.text + ".0"))}
    ;

r11
    : expression -> ^(EXPR expression)
    | -> EXPR
    ;

r12
    : ID (',' ID)* -> ID+
    ;

r13
    : type ID (',' ID)* ';' -> ^(type ID+)
    ;

r14
    :   expression? statement* type+
        -> ^(EXPR expression? statement* type+)
    ;

r15
    : INT -> INT INT
    ;

r16
    : 'int' ID (',' ID)* -> ^('int' ID)+
    ;

r17
    : 'for' '(' start=statement ';' expression ';' next=statement ')' statement
        -> ^('for' $start expression $next statement)
    ;

r18
    : t='for' -> ^(BLOCK)
    ;

r19
    : t='for' -> ^(BLOCK[$t])
    ;

r20
    : t='for' -> ^(BLOCK[$t,"FOR"])
    ;

r21
    : t='for' -> BLOCK
    ;

r22
    : t='for' -> BLOCK[$t]
    ;

r23
    : t='for' -> BLOCK[$t,"FOR"]
    ;

r24
    : r=statement expression -> ^($r expression)
    ;

r25
    : r+=statement (',' r+=statement)+ expression -> ^($r expression)
    ;

r26
    : r+=statement (',' r+=statement)+ -> ^(BLOCK $r+)
    ;

r27
    : r=statement expression -> ^($r ^($r expression))
    ;

r28
    : ('foo28a'|'foo28b') ->
    ;

r29
    : (r+=statement)* -> ^(BLOCK $r+)
    ;

r30
    : statement* -> ^(BLOCK statement?)
    ;

r31
    : modifier type ID ('=' expression)? ';'
        -> {this.flag === 0}? ^(VARDEF ID modifier* type expression?)
        -> {this.flag === 1}? ^(VARIABLE ID modifier* type expression?)
        ->                   ^(FIELD ID modifier* type expression?)
    ;

r32[which]
  : ID INT -> {which==1}? ID
           -> {which==2}? INT
           -> // yield nothing as else-clause
  ;

r33
    :   modifiers! statement
    ;

r34
    :   modifiers! r34a[$modifiers.tree]
    //|   modifiers! r33b[$modifiers.tree]
    ;

r34a[mod]
    :   'class' ID ('extends' sup=type)?
        ( 'implements' i+=type (',' i+=type)*)?
        '{' statement* '}'
        -> ^('class' ID {$mod} ^('extends' $sup)? ^('implements' $i+)? statement* )
    ;

r35
    : '{' 'extends' (sup=type)? '}'
        ->  ^('extends' $sup)?
    ;

r36
    : 'if' '(' expression ')' s1=statement
        ( 'else' s2=statement -> ^('if' ^(EXPR expression) $s1 $s2)
        |                     -> ^('if' ^(EXPR expression) $s1)
        )
    ;

r37
    : (INT -> INT) ('+' i=INT -> ^('+' $r37 $i) )* 
    ;

r38
    : INT ('+'^ INT)*
    ;

r39
    : (primary->primary) // set return tree to just primary
        ( '(' arg=expression ')'
            -> ^(CALL $r39 $arg)
        | '[' ie=expression ']'
            -> ^(INDEX $r39 $ie)
        | '.' p=primary
            -> ^(FIELDACCESS $r39 $p)
        )*
    ;

r40
    : (INT -> INT) ( ('+' i+=INT)* -> ^('+' $r40 $i*) ) ';'
    ;

r41
    : (INT -> INT) ( ('+' i=INT) -> ^($i $r41) )* ';'
    ;

r42
    : ids+=ID (','! ids+=ID)*
    ;

r43 returns [res]
    : ids+=ID! (','! ids+=ID!)* {$res = org.antlr.lang.map($ids, function(id) { return id.getText(); });}
    ;

r44
    : ids+=ID^ (','! ids+=ID^)*
    ;

r45
    : primary^
    ;

r46 returns [res]
    : ids+=primary! (','! ids+=primary!)* {$res = org.antlr.lang.map($ids, function(id) { return id.getText(); });}
    ;

r47
    : ids+=primary (','! ids+=primary)*
    ;

r48
    : ids+=. (','! ids+=.)*
    ;

r49
    : .^ ID
    ;

r50
    : ID 
        -> ^({new org.antlr.runtime.tree.CommonTree(new org.antlr.runtime.CommonToken(FLOAT, "1.0"))} ID)
    ;

/** templates tested:
    tokenLabelPropertyRef_tree
*/
r51 returns [res]
    : ID t=ID ID
        { $res = $t.tree; }
    ;

/** templates tested:
    rulePropertyRef_tree
*/
r52 returns [res]
@after {
    $res = $tree;
}
    : ID
    ;

/** templates tested:
    ruleLabelPropertyRef_tree
*/
r53 returns [res]
    : t=primary
        { $res = $t.tree; }
    ;

/** templates tested:
    ruleSetPropertyRef_tree
*/
r54 returns [res]
@after {
    $tree = $t.tree;;
}
    : ID t=expression ID
    ;

/** backtracking */
r55
options { backtrack=true; k=1; }
    : (modifier+ INT)=> modifier+ expression
    | modifier+ statement
    ;


/** templates tested:
    rewriteTokenRef with len(args)>0
*/
r56
    : t=ID* -> ID[$t,'foo']
    ;

/** templates tested:
    rewriteTokenRefRoot with len(args)>0
*/
r57
    : t=ID* -> ^(ID[$t,'foo'])
    ;

/** templates tested:
    ???
*/
r58
    : ({new org.antlr.runtime.tree.CommonTree(new org.antlr.runtime.CommonToken(FLOAT, "2.0"))})^
    ;

/** templates tested:
    rewriteTokenListLabelRefRoot
*/
r59
    : (t+=ID)+ statement -> ^($t statement)+
    ;

primary
    : ID
    ;

expression
    : r1
    ;

statement
    : 'fooze'
    | 'fooze2'
    ;

modifiers
    : modifier+
    ;

modifier
    : 'public'
    | 'private'
    ;

type
    : 'int'
    | 'bool'
    ;

ID : 'a'..'z' + ;
INT : '0'..'9' +;
WS: (' ' | '\n' | '\t')+ {$channel = HIDDEN;};

