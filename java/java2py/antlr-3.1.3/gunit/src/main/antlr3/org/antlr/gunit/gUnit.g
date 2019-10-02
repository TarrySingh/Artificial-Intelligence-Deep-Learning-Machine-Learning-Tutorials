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
grammar gUnit;
options {language=Java;}
tokens {
	OK = 'OK';
	FAIL = 'FAIL';
	DOC_COMMENT;
}
@header {package org.antlr.gunit;}
@lexer::header {
package org.antlr.gunit;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
}
@members {
public GrammarInfo grammarInfo;
public gUnitParser(TokenStream input, GrammarInfo grammarInfo) {
	super(input);
	this.grammarInfo = grammarInfo;
}
}

gUnitDef:	'gunit' g1=id ('walks' g2=id)? ';' 
		{
		if ( $g2.text!=null ) {
			grammarInfo.setGrammarName($g2.text);
			grammarInfo.setTreeGrammarName($g1.text);
		}
		else {
			grammarInfo.setGrammarName($g1.text);
		}
		}
		header? testsuite*
	;

header	:	'@header' ACTION
		{
		int pos1, pos2;
		if ( (pos1=$ACTION.text.indexOf("package"))!=-1 && (pos2=$ACTION.text.indexOf(';'))!=-1 ) {
			grammarInfo.setHeader($ACTION.text.substring(pos1+8, pos2).trim());	// substring the package path
		}
		else {
			System.err.println("error(line "+$ACTION.getLine()+"): invalid header");
		}
		}
	;
		
testsuite	// gUnit test suite based on individual rule
scope {
boolean isLexicalRule;
}
@init {
gUnitTestSuite ts = null;
$testsuite::isLexicalRule = false;
}
	:	(	r1=RULE_REF ('walks' r2=RULE_REF)? 
			{
			if ( $r2==null ) ts = new gUnitTestSuite($r1.text);
			else ts = new gUnitTestSuite($r1.text, $r2.text);
			}
		|	t=TOKEN_REF 
			{
			ts = new gUnitTestSuite();
			ts.setLexicalRuleName($t.text);
			$testsuite::isLexicalRule = true;
			}
		)
		':'
		testcase[ts]+ {grammarInfo.addRuleTestSuite(ts);}
	;

// TODO : currently gUnit just ignores illegal test for lexer rule, but should also emit a reminding message
testcase[gUnitTestSuite ts]	// individual test within a (rule)testsuite
	:	input expect {$ts.addTestCase($input.in, $expect.out);}
	;

input returns [gUnitTestInput in]
@init {
String testInput = null;
boolean inputIsFile = false;
int line = -1;
}
@after {
in = new gUnitTestInput(testInput, inputIsFile, line);
}
	:	STRING 
		{
		testInput = $STRING.text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
		.replace("\\b", "\b").replace("\\f", "\f").replace("\\\"", "\"").replace("\\'", "\'").replace("\\\\", "\\");
		line = $STRING.line;
		}
	|	ML_STRING
		{
		testInput = $ML_STRING.text;
		line = $ML_STRING.line;
		}
	|	file
		{
		testInput = $file.text;
		inputIsFile = true;
		line = $file.line;
		}
	;
	
expect returns [AbstractTest out]
	:	OK {$out = new BooleanTest(true);}
	|	FAIL {$out = new BooleanTest(false);}
	|	'returns' RETVAL {if ( !$testsuite::isLexicalRule ) $out = new ReturnTest($RETVAL);}
	|	'->' output {if ( !$testsuite::isLexicalRule ) $out = new OutputTest($output.token);}
	;

output returns [Token token]
	:	STRING 
		{
		$STRING.setText($STRING.text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
		.replace("\\b", "\b").replace("\\f", "\f").replace("\\\"", "\"").replace("\\'", "\'").replace("\\\\", "\\"));
		$token = $STRING;
		}
	|	ML_STRING {$token = $ML_STRING;}
	|	AST {$token = $AST;}
	|	ACTION {$token = $ACTION;}
	;

file returns [int line]	
	:	id EXT? {$line = $id.line;}
	;

id returns [int line] 
	:	TOKEN_REF {$line = $TOKEN_REF.line;}
	|	RULE_REF {$line = $RULE_REF.line;}
	;

// L E X I C A L   R U L E S

SL_COMMENT
 	:	'//' ~('\r'|'\n')* '\r'? '\n' {$channel=HIDDEN;}
	;

ML_COMMENT
	:	'/*' {$channel=HIDDEN;} .* '*/'
	;

STRING	:	'"' ( ESC | ~('\\'|'"') )* '"' {setText(getText().substring(1, getText().length()-1));}
	;

ML_STRING
	:	{// we need to determine the number of spaces or tabs (indentation) for multi-line input
		StringBuffer buf = new StringBuffer();
		int i = -1;
		int c = input.LA(-1);
		while ( c==' ' || c=='\t' ) {
			buf.append((char)c);
			c = input.LA(--i);
		}
		String indentation = buf.reverse().toString();
		}
		'<<' .* '>>' 
		{// also determine the appropriate newline separator and get info of the first and last 2 characters (exclude '<<' and '>>')
		String newline = System.getProperty("line.separator");
		String front, end;
		int oldFrontIndex = 2;
		int oldEndIndex = getText().length()-2;
		int newFrontIndex, newEndIndex;
		if ( newline.length()==1 ) {
			front = getText().substring(2, 3);
			end = getText().substring(getText().length()-3, getText().length()-2);
			newFrontIndex = 3;
			newEndIndex = getText().length()-3;
		}
		else {// must be 2, e.g. Windows System which uses \r\n as a line separator
			front = getText().substring(2, 4);
			end = getText().substring(getText().length()-4, getText().length()-2);
			newFrontIndex = 4;
			newEndIndex = getText().length()-4;
		}
		// strip unwanted characters, e.g. '<<' (including a newline after it) or '>>'  (including a newline before it)
		String temp = null;
		if ( front.equals(newline) && end.equals(newline) ) {
			// need to handle the special case: <<\n>> or <<\r\n>>
			if ( newline.length()==1 && getText().length()==5 ) temp = "";
			else if ( newline.length()==2 && getText().length()==6 ) temp = "";
			else temp = getText().substring(newFrontIndex, newEndIndex);
		}
		else if ( front.equals(newline) ) {
			temp = getText().substring(newFrontIndex, oldEndIndex);
		}
		else if ( end.equals(newline) ) {
			temp = getText().substring(oldFrontIndex, newEndIndex);
		}
		else {
			temp = getText().substring(oldFrontIndex, oldEndIndex);
		}
		// finally we need to prpcess the indentation line by line
		BufferedReader bufReader = new BufferedReader(new StringReader(temp));
		buf = new StringBuffer();
		String line = null;
		int count = 0;
		try {
			while((line = bufReader.readLine()) != null) {
				if ( line.startsWith(indentation) ) line = line.substring(indentation.length());
				if ( count>0 ) buf.append(newline);
				buf.append(line);
				count++;
			}
			setText(buf.toString());
		}
		catch (IOException ioe) {
			setText(temp);
		}
		}
	;

TOKEN_REF
	:	'A'..'Z' ('a'..'z'|'A'..'Z'|'_'|'0'..'9')*
	;

RULE_REF
	:	'a'..'z' ('a'..'z'|'A'..'Z'|'_'|'0'..'9')*
	;

EXT	:	'.'('a'..'z'|'A'..'Z'|'0'..'9')+;

RETVAL	:	NESTED_RETVAL {setText(getText().substring(1, getText().length()-1));}
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
	:	NESTED_ACTION {setText(getText().substring(1, getText().length()-1));}
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
