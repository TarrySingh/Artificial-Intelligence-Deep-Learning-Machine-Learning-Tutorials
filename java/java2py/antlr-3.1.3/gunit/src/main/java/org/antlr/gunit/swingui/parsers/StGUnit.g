/*
[The "BSD licence"]
Copyright (c) 2007-2008 Leon Jen-Yuan Su
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

grammar StGUnit;

options {language=Java;}

tokens {
	OK = 'OK';
	FAIL = 'FAIL';
	DOC_COMMENT;
}

@header {
package org.antlr.gunit.swingui.parsers;
import org.antlr.gunit.swingui.model.*;
}

@lexer::header {package org.antlr.gunit.swingui;}

@members {
public Translator.TestSuiteAdapter adapter ;;
}

gUnitDef
	:	'gunit' name=id {adapter.setGrammarName($name.text);}
	    ('walks' id)? ';' 
		header? suite*
	;

header
	:	'@header' ACTION
	;
		
suite
	:	(	parserRule=RULE_REF ('walks' RULE_REF)? 
	        {adapter.startRule($parserRule.text);}
		|	lexerRule=TOKEN_REF 
			{adapter.startRule($lexerRule.text);}
		)
		':'
		test+
		{adapter.endRule();}
	;

test
	:	input expect
		{adapter.addTestCase($input.in, $expect.out);}
	;
	
expect returns [ITestCaseOutput out]
	:	OK			{$out = adapter.createBoolOutput(true);}
	|	FAIL		{$out = adapter.createBoolOutput(false);}
	|	'returns' RETVAL {$out = adapter.createReturnOutput($RETVAL.text);}
	|	'->' output {$out = adapter.createStdOutput($output.text);}
	|	'->' AST	{$out = adapter.createAstOutput($AST.text);}
	;

input returns [ITestCaseInput in]
	:	STRING 		{$in = adapter.createStringInput($STRING.text);}
	|	ML_STRING	{$in = adapter.createMultiInput($ML_STRING.text);}
	|	fileInput	{$in = adapter.createFileInput($fileInput.path);}
	;

output
	:	STRING
	|	ML_STRING
	|	ACTION
	;
	
fileInput returns [String path]
	:	id {$path = $id.text;} (EXT {$path += $EXT.text;})? 
	;

id 	:	TOKEN_REF
	|	RULE_REF
	;

// L E X I C A L   R U L E S

SL_COMMENT
 	:	'//' ~('\r'|'\n')* '\r'? '\n' {$channel=HIDDEN;}
	;

ML_COMMENT
	:	'/*' {$channel=HIDDEN;} .* '*/'
	;

STRING
	:	'"' ( ESC | ~('\\'|'"') )* '"'
	;

ML_STRING
	:	'<<' .* '>>' 
	;

TOKEN_REF
	:	'A'..'Z' ('a'..'z'|'A'..'Z'|'_'|'0'..'9')*
	;

RULE_REF
	:	'a'..'z' ('a'..'z'|'A'..'Z'|'_'|'0'..'9')*
	;

EXT	:	'.'('a'..'z'|'A'..'Z'|'0'..'9')+;

RETVAL	:	NESTED_RETVAL
	;

fragment
NESTED_RETVAL :
	'['
	(	options {greedy=false;}
	:	NESTED_RETVAL
	|	.
	)*
	']'
	;

AST	:	NESTED_AST (' '? NESTED_AST)*;

fragment
NESTED_AST :
	'('
	(	options {greedy=false;}
	:	NESTED_AST
	|	.
	)*
	')'
	;

ACTION
	:	NESTED_ACTION
	;

fragment
NESTED_ACTION :
	'{'
	(	options {greedy=false; k=3;}
	:	NESTED_ACTION
	|	STRING_LITERAL
	|	CHAR_LITERAL
	|	.
	)*
	'}'
	;

fragment
CHAR_LITERAL
	:	'\'' ( ESC | ~('\''|'\\') ) '\''
	;

fragment
STRING_LITERAL
	:	'"' ( ESC | ~('\\'|'"') )* '"'
	;

fragment
ESC	:	'\\'
		(	'n'
		|	'r'
		|	't'
		|	'b'
		|	'f'
		|	'"'
		|	'\''
		|	'\\'
		|	'>'
		|	'u' XDIGIT XDIGIT XDIGIT XDIGIT
		|	. // unknown, leave as it is
		)
	;
	
fragment
XDIGIT :
		'0' .. '9'
	|	'a' .. 'f'
	|	'A' .. 'F'
	;

WS	:	(	' '
		|	'\t'
		|	'\r'? '\n'
		)+
		{$channel=HIDDEN;}
	;
