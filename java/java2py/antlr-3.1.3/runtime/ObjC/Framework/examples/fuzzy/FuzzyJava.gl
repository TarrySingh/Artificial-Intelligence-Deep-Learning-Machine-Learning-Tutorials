lexer grammar FuzzyJava;
options {filter=true; language=ObjC;}

IMPORT
	:	'import' WS name=QIDStar WS? ';'
	;
	
/** Avoids having "return foo;" match as a field */
RETURN
	:	'return' (options {greedy=false;}:.)* ';'
	;

CLASS
	:	'class' WS name=ID WS? ('extends' WS QID WS?)?
		('implements' WS QID WS? (',' WS? QID WS?)*)? '{'
        {NSLog(@"found class \%@", $name.text);}
	;
	
METHOD
    :   TYPE WS name=ID WS? '(' ( ARG WS? (',' WS? ARG WS?)* )? ')' WS? 
       ('throws' WS QID WS? (',' WS? QID WS?)*)? '{'
        {NSLog(@"found method \%@", $name.text);}
    ;

FIELD
    :   TYPE WS name=ID '[]'? WS? (';'|'=')
        {NSLog(@"found var \%@", $name.text);}
    ;

STAT:	('if'|'while'|'switch'|'for') WS? '(' ;
	
CALL
    :   name=QID WS? '('
        {/*ignore if this/super */ NSLog(@"found call \%@",$name.text);}
    ;

COMMENT
    :   '/*' (options {greedy=false;} : . )* '*/'
        {NSLog(@"found comment \%@", [self text]);}
    ;

SL_COMMENT
    :   '//' (options {greedy=false;} : . )* '\n'
        {NSLog(@"found // comment \%@", [self text]);}
    ;
	
STRING
	:	'"' (options {greedy=false;}: ESC | .)* '"'
	;

CHAR
	:	'\'' (options {greedy=false;}: ESC | .)* '\''
	;

WS  :   (' '|'\t'|'\n')+
    ;

fragment
QID :	ID ('.' ID)*
	;
	
/** QID cannot see beyond end of token so using QID '.*'? somewhere won't
 *  ever match since k=1 lookahead in the QID loop of '.' will make it loop.
 *  I made this rule to compensate.
 */
fragment
QIDStar
	:	ID ('.' ID)* '.*'?
	;

fragment
TYPE:   QID '[]'?
    ;
    
fragment
ARG :   TYPE WS ID
    ;

fragment
ID  :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'_'|'0'..'9')*
    ;

fragment
ESC	:	'\\' ('"'|'\''|'\\')
	;

/* Must peel these off first so they are not IDs.  Must also distinguish
 *  between return foo; and typename foo;
KEYWORD
@init {NSLog("KEYWORD: "+(char)input.LA(1));}
	:	'abstract'
	|	'assert'
	|	'boolean'
	|	'break'
	|	'byte'
	|	'case'
	|	'catch'
	|	'char'
	|	'class'
	|	'const'
	|	'continue'
	|	'default'
	|	'do'
	|	'double'
	|	'else'
	|	'enum'
	|	'extends'
	|	'false'
	|	'final'
	|	'finally'
	|	'float'
	|	'for'
	|	'goto'
	|	'if'
	|	'implements'
	|	'import'
	|	'instanceof'
	|	'int'
	|	'interface'
	|	'long'
	|	'native'
	|	'new'
	|	'null'
	|	'package'
	|	'private'
	|	'protected'
	|	'public'
	|	'return' {NSLog("found return");}
	|	'short'
	|	'static'
	|	'strictfp'
	|	'super'
	|	'switch'
	|	'synchronized'
	|	'this'
	|	'throw'
	|	'throws'
	|	'transient'
	|	'true'
	|	'try'
	|	'void'
	|	'volatile'
	|	'while'
	;
*/

/*
MAIN
//options {k=2;}
	:	(KEYWORD)=>KEYWORD
	|	(FIELD)=>FIELD
	|	(METHOD)=>METHOD
	|	STRING
	|	COMMENT
	|	SL_COMMENT
	|	WS
//	|	.
	;
	*/

