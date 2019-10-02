grammar SimpleC;

options {
	language=ObjC;
}

program
    :   declaration+
    ;

/** In this rule, the functionHeader left prefix on the last two
 *  alternatives is not LL(k) for a fixed k.  However, it is
 *  LL(*).  The LL(*) algorithm simply scans ahead until it sees
 *  either the ';' or the '{' of the block and then it picks
 *  the appropriate alternative.  Lookhead can be arbitrarily
 *  long in theory, but is <=10 in most cases.  Works great.
 *  Use ANTLRWorks to see the lookahead use (step by Location)
 *  and look for blue tokens in the input window pane. :)
 */
declaration
    :   variable
    |   functionHeader ';'
	{NSLog(@"\%@ is a declaration", $functionHeader.name);}
    |   functionHeader block
	{NSLog(@"\%@ is a definition", $functionHeader.name);}
    ;

variable
    :   type declarator ';'
    ;

declarator
    :   ID 
    ;

functionHeader returns [NSString* name]
@init {
    $name=nil; // for now you must init here rather than in 'returns'
}
    :   type ID '(' ( formalParameter ( ',' formalParameter )* )? ')'
	{$name = $ID.text; }
    ;

formalParameter
    :   type declarator        
    ;

type
    :   'int'   
    |   'char'  
    |   'void'
    |   ID        
    ;

block
    :   '{'
            variable*
            stat*
        '}'
    ;

stat: forStat
    | expr ';'      
    | block
    | assignStat ';'
    | ';'
    ;

forStat
    :   'for' '(' assignStat ';' expr ';' assignStat ')' block        
    ;

assignStat
    :   ID '=' expr        
    ;

expr:   condExpr
    ;

condExpr
    :   aexpr ( ('==' | '<') aexpr )?
    ;

aexpr
    :   atom ( '+' atom )*
    ;

atom
    : ID      
    | INT      
    | '(' expr ')'
    ; 

ID  :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
    ;

INT :	('0'..'9')+
    ;

WS  :   (   ' '
        |   '\t'
        |   '\r'
        |   '\n'
        )+
        { $channel=99; }
    ;    
