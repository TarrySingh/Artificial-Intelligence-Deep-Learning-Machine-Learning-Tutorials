/*
 [The 'BSD licence']
 Copyright (c) 2004 Terence Parr and Loring Craymer
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** Python 2.3.3 Grammar
 *
 *  Terence Parr and Loring Craymer
 *  February 2004
 *
 *  Converted to ANTLR v3 November 2005 by Terence Parr.
 *
 *  This grammar was derived automatically from the Python 2.3.3
 *  parser grammar to get a syntactically correct ANTLR grammar
 *  for Python.  Then Terence hand tweaked it to be semantically
 *  correct; i.e., removed lookahead issues etc...  It is LL(1)
 *  except for the (sometimes optional) trailing commas and semi-colons.
 *  It needs two symbols of lookahead in this case.
 *
 *  Starting with Loring's preliminary lexer for Python, I modified it
 *  to do my version of the whole nasty INDENT/DEDENT issue just so I
 *  could understand the problem better.  This grammar requires
 *  PythonTokenStream.java to work.  Also I used some rules from the
 *  semi-formal grammar on the web for Python (automatically
 *  translated to ANTLR format by an ANTLR grammar, naturally <grin>).
 *  The lexical rules for python are particularly nasty and it took me
 *  a long time to get it 'right'; i.e., think about it in the proper
 *  way.  Resist changing the lexer unless you've used ANTLR a lot. ;)
 *
 *  I (Terence) tested this by running it on the jython-2.1/Lib
 *  directory of 40k lines of Python.
 *
 *  REQUIRES ANTLR v3
 */
grammar Python;
options {language=JavaScript;}

tokens {
    INDENT;
    DEDENT;
}

@lexer::members {
/** Handles context-sensitive lexing of implicit line joining such as
 *  the case where newline is ignored in cases like this:
 *  a = [3,
 *       4]
 */
	this.implicitLineJoiningLevel= 0;
	this.startPos = -1;
}

single_input
    : NEWLINE
	| simple_stmt
	| compound_stmt NEWLINE
	;

file_input
    :   (NEWLINE | stmt)*
	;

eval_input
    :   (NEWLINE)* testlist (NEWLINE)*
	;

funcdef
    :   'def' NAME parameters COLON suite
	{xlog("found method def "+$NAME.text);}
	;

parameters
    :   LPAREN (varargslist)? RPAREN
	;

varargslist
    :   defparameter (options {greedy=true;}:COMMA defparameter)*
        (COMMA
            ( STAR NAME (COMMA DOUBLESTAR NAME)?
            | DOUBLESTAR NAME
            )?
        )?
    |   STAR NAME (COMMA DOUBLESTAR NAME)?
    |   DOUBLESTAR NAME
    ;

defparameter
    :   fpdef (ASSIGN test)?
    ;

fpdef
    :   NAME
	|   LPAREN fplist RPAREN
	;

fplist
    :   fpdef (options {greedy=true;}:COMMA fpdef)* (COMMA)?
	;


stmt: simple_stmt
	| compound_stmt
	;

simple_stmt
    :   small_stmt (options {greedy=true;}:SEMI small_stmt)* (SEMI)? NEWLINE
	;

small_stmt: expr_stmt
	| print_stmt
	| del_stmt
	| pass_stmt
	| flow_stmt
	| import_stmt
	| global_stmt
	| exec_stmt
	| assert_stmt
	;

expr_stmt
	:	testlist
		(	augassign testlist
		|	(ASSIGN testlist)+
		)?
	;

augassign
    : PLUSEQUAL
	| MINUSEQUAL
	| STAREQUAL
	| SLASHEQUAL
	| PERCENTEQUAL
	| AMPEREQUAL
	| VBAREQUAL
	| CIRCUMFLEXEQUAL
	| LEFTSHIFTEQUAL
	| RIGHTSHIFTEQUAL
	| DOUBLESTAREQUAL
	| DOUBLESLASHEQUAL
	;

print_stmt:
        'print'
        (   testlist
        |   RIGHTSHIFT testlist
        )?
	;

del_stmt: 'del' exprlist
	;

pass_stmt: 'pass'
	;

flow_stmt: break_stmt
	| continue_stmt
	| return_stmt
	| raise_stmt
	| yield_stmt
	;

break_stmt: 'break'
	;

continue_stmt: 'continue'
	;

return_stmt: 'return' (testlist)?
	;

yield_stmt: 'yield' testlist
	;

raise_stmt: 'raise' (test (COMMA test (COMMA test)?)?)?
	;

import_stmt
    :   'import' dotted_as_name (COMMA dotted_as_name)*
	|   'from' dotted_name 'import'
        (STAR | import_as_name (COMMA import_as_name)*)
	;

import_as_name
    :   NAME (NAME NAME)?
	;

dotted_as_name: dotted_name (NAME NAME)?
	;

dotted_name: NAME (DOT NAME)*
	;

global_stmt: 'global' NAME (COMMA NAME)*
	;

exec_stmt: 'exec' expr ('in' test (COMMA test)?)?
	;

assert_stmt: 'assert' test (COMMA test)?
	;


compound_stmt: if_stmt
	| while_stmt
	| for_stmt
	| try_stmt
	| funcdef
	| classdef
	;

if_stmt: 'if' test COLON suite ('elif' test COLON suite)* ('else' COLON suite)?
	;

while_stmt: 'while' test COLON suite ('else' COLON suite)?
	;

for_stmt: 'for' exprlist 'in' testlist COLON suite ('else' COLON suite)?
	;

try_stmt
    :   'try' COLON suite
        (   (except_clause COLON suite)+ ('else' COLON suite)?
        |   'finally' COLON suite
        )
	;

except_clause: 'except' (test (COMMA test)?)?
	;

suite: simple_stmt
	| NEWLINE INDENT (stmt)+ DEDENT
	;


test: and_test ('or' and_test)*
	| lambdef
	;

and_test
	: not_test ('and' not_test)*
	;

not_test
	: 'not' not_test
	| comparison
	;

comparison: expr (comp_op expr)*
	;

comp_op: LESS
	|GREATER
	|EQUAL
	|GREATEREQUAL
	|LESSEQUAL
	|ALT_NOTEQUAL
	|NOTEQUAL
	|'in'
	|'not' 'in'
	|'is'
	|'is' 'not'
	;

expr: xor_expr (VBAR xor_expr)*
	;

xor_expr: and_expr (CIRCUMFLEX and_expr)*
	;

and_expr: shift_expr (AMPER shift_expr)*
	;

shift_expr: arith_expr ((LEFTSHIFT|RIGHTSHIFT) arith_expr)*
	;

arith_expr: term ((PLUS|MINUS) term)*
	;

term: factor ((STAR | SLASH | PERCENT | DOUBLESLASH ) factor)*
	;

factor
	: (PLUS|MINUS|TILDE) factor
	| power
	;

power
	:   atom (trailer)* (options {greedy=true;}:DOUBLESTAR factor)?
	;

atom: LPAREN (testlist)? RPAREN
	| LBRACK (listmaker)? RBRACK
	| LCURLY (dictmaker)? RCURLY
	| BACKQUOTE testlist BACKQUOTE
	| NAME
	| INT
    | LONGINT
    | FLOAT
    | COMPLEX
	| (STRING)+
	;

listmaker: test ( list_for | (options {greedy=true;}:COMMA test)* ) (COMMA)?
	;

lambdef: 'lambda' (varargslist)? COLON test
	;

trailer: LPAREN (arglist)? RPAREN
	| LBRACK subscriptlist RBRACK
	| DOT NAME
	;

subscriptlist
    :   subscript (options {greedy=true;}:COMMA subscript)* (COMMA)?
	;

subscript
	: DOT DOT DOT
    | test (COLON (test)? (sliceop)?)?
    | COLON (test)? (sliceop)?
    ;

sliceop: COLON (test)?
	;

exprlist
    :   expr (options {k=2;}:COMMA expr)* (COMMA)?
	;

testlist
    :   test (options {k=2;}: COMMA test)* (COMMA)?
    ;

dictmaker
    :   test COLON test
        (options {k=2;}:COMMA test COLON test)* (COMMA)?
    ;

classdef: 'class' NAME (LPAREN testlist RPAREN)? COLON suite
	{xlog("found class def "+$NAME.text);}
	;

arglist: argument (COMMA argument)*
        ( COMMA
          ( STAR test (COMMA DOUBLESTAR test)?
          | DOUBLESTAR test
          )?
        )?
    |   STAR test (COMMA DOUBLESTAR test)?
    |   DOUBLESTAR test
    ;

argument : test (ASSIGN test)?
         ;

list_iter: list_for
	| list_if
	;

list_for: 'for' exprlist 'in' testlist (list_iter)?
	;

list_if: 'if' test (list_iter)?
	;

LPAREN	: '(' {this.implicitLineJoiningLevel++;} ;

RPAREN	: ')' {this.implicitLineJoiningLevel--;} ;

LBRACK	: '[' {this.implicitLineJoiningLevel++;} ;

RBRACK	: ']' {this.implicitLineJoiningLevel--;} ;

COLON 	: ':' ;

COMMA	: ',' ;

SEMI	: ';' ;

PLUS	: '+' ;

MINUS	: '-' ;

STAR	: '*' ;

SLASH	: '/' ;

VBAR	: '|' ;

AMPER	: '&' ;

LESS	: '<' ;

GREATER	: '>' ;

ASSIGN	: '=' ;

PERCENT	: '%' ;

BACKQUOTE	: '`' ;

LCURLY	: '{' {this.implicitLineJoiningLevel++;} ;

RCURLY	: '}' {this.implicitLineJoiningLevel--;} ;

CIRCUMFLEX	: '^' ;

TILDE	: '~' ;

EQUAL	: '==' ;

NOTEQUAL	: '!=' ;

ALT_NOTEQUAL: '<>' ;

LESSEQUAL	: '<=' ;

LEFTSHIFT	: '<<' ;

GREATEREQUAL	: '>=' ;

RIGHTSHIFT	: '>>' ;

PLUSEQUAL	: '+=' ;

MINUSEQUAL	: '-=' ;

DOUBLESTAR	: '**' ;

STAREQUAL	: '*=' ;

DOUBLESLASH	: '//' ;

SLASHEQUAL	: '/=' ;

VBAREQUAL	: '|=' ;

PERCENTEQUAL	: '%=' ;

AMPEREQUAL	: '&=' ;

CIRCUMFLEXEQUAL	: '^=' ;

LEFTSHIFTEQUAL	: '<<=' ;

RIGHTSHIFTEQUAL	: '>>=' ;

DOUBLESTAREQUAL	: '**=' ;

DOUBLESLASHEQUAL	: '//=' ;

DOT : '.' ;

FLOAT
	:	'.' DIGITS (Exponent)?
    |   DIGITS ('.' (DIGITS (Exponent)?)? | Exponent)
    ;

LONGINT
    :   INT ('l'|'L')
    ;

fragment
Exponent
	:	('e' | 'E') ( '+' | '-' )? DIGITS
	;

INT :   // Hex
        '0' ('x' | 'X') ( '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' )+
        ('l' | 'L')?
    |   // Octal
        '0' DIGITS*
    |   '1'..'9' DIGITS*
    ;

COMPLEX
    :   INT ('j'|'J')
    |   FLOAT ('j'|'J')
    ;

fragment
DIGITS : ( '0' .. '9' )+ ;

NAME:	( 'a' .. 'z' | 'A' .. 'Z' | '_')
        ( 'a' .. 'z' | 'A' .. 'Z' | '_' | '0' .. '9' )*
    ;

/** Match various string types.  Note that greedy=false implies '''
 *  should make us exit loop not continue.
 */
STRING
    :   ('r'|'u'|'ur')?
        (   '\'\'\'' (options {greedy=false;}:.)* '\'\'\''
        |   '"""' (options {greedy=false;}:.)* '"""'
        |   '"' (ESC|~('\\'|'\n'|'"'))* '"'
        |   '\'' (ESC|~('\\'|'\n'|'\''))* '\''
        )
	;

fragment
ESC
	:	'\\' .
	;

/** Consume a newline and any whitespace at start of next line */
CONTINUED_LINE
	:	'\\' ('\r')? '\n' (' '|'\t')* { $channel=HIDDEN; }
	;

/** Treat a sequence of blank lines as a single blank line.  If
 *  nested within a (..), {..}, or [..], then ignore newlines.
 *  If the first newline starts in column one, they are to be ignored.
 */
NEWLINE
    :   (('\r')? '\n' )+
        {if ( this.startPos==0 || this.implicitLineJoiningLevel>0 )
            $channel=HIDDEN;
        }
    ;

WS	:	{this.startPos>0}?=> (' '|'\t')+ {$channel=HIDDEN;}
	;
	
/** Grab everything before a real symbol.  Then if newline, kill it
 *  as this is a blank line.  If whitespace followed by comment, kill it
 *  as it's a comment on a line by itself.
 *
 *  Ignore leading whitespace when nested in [..], (..), {..}.
 */
LEADING_WS
@init {
    var spaces = 0;
}
    :   {this.startPos==0}?=>
    	(   {this.implicitLineJoiningLevel>0}? ( ' ' | '\t' )+ {$channel=HIDDEN;}
       	|	( 	' '  { spaces++; }
        	|	'\t' { spaces += 8; spaces -= (spaces \% 8); }
       		)+
        	{
            // make a string of n spaces where n is column number - 1
            var indentation = new Array(spaces);
            for (var i=0; i<spaces; i++) {
                indentation[i] = ' ';
            }
            var s = indentation.join("");
            this.emit(new org.antlr.runtime.CommonToken(this.LEADING_WS,s));
        	}
        	// kill trailing newline if present and then ignore
        	( ('\r')? '\n' {if (this.state.token!=null) this.state.token.setChannel(HIDDEN); else $channel=HIDDEN;})*
           // {this.token.setChannel(99); }
        )
    ;

/** Comments not on line by themselves are turned into newlines.

    b = a # end of line comment

    or

    a = [1, # weird
         2]

    This rule is invoked directly by nextToken when the comment is in
    first column or when comment is on end of nonwhitespace line.

	Only match \n here if we didn't start on left edge; let NEWLINE return that.
	Kill if newlines if we live on a line by ourselves
	
	Consume any leading whitespace if it starts on left edge.
 */
COMMENT
@init {
    $channel=HIDDEN;
}
    :	{this.startPos==0}?=> (' '|'\t')* '#' (~'\n')* '\n'+
    |	{this.startPos>0}?=> '#' (~'\n')* // let NEWLINE handle \n unless char pos==0 for '#'
    ;
