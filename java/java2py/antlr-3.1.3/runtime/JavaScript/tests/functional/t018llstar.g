grammar t018llstar;

options {
    language = JavaScript;
}

@members {
this.olog = [];
this.output = function(msg) {
    this.olog.push(msg);
};
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
	{this.output($functionHeader.name+" is a declaration");}
    |   functionHeader block
	{this.output($functionHeader.name+" is a definition");}
    ;

variable
    :   type declarator ';'
    ;

declarator
    :   ID 
    ;

functionHeader returns [name]
    :   type ID '(' ( formalParameter ( ',' formalParameter )* )? ')'
	{$name = $ID.text}
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
        {$channel=HIDDEN}
    ;    
