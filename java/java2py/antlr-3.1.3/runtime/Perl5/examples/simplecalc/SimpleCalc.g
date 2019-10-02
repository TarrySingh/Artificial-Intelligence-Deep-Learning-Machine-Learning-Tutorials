grammar SimpleCalc;
options { language = Perl5; }

tokens {
    PLUS  = '+' ;
    MINUS = '-' ;
    MULT  = '*' ;
    DIV   = '/' ;
}

/*------------------------------------------------------------------
 * PARSER RULES
 *------------------------------------------------------------------*/

expr    : term ( ( PLUS | MINUS )  term )* ;

term    : factor ( ( MULT | DIV ) factor )* ;

factor  : NUMBER ;

/*------------------------------------------------------------------
 * LEXER RULES
 *------------------------------------------------------------------*/

NUMBER : (DIGIT)+ ;

WHITESPACE : ( '\t' | ' ' | '\r' | '\n'| '\u000C' )+ { $channel = $self->HIDDEN; } ;

fragment DIGIT : '0'..'9' ;
