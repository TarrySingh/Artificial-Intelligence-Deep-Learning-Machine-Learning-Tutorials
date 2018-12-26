/*
 * Javadoc.g
 * Copyright (c) 2007 David Holroyd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

grammar JavaDoc;


options {
	k=3;
	output=AST;
    language=Python;
	ASTLabelType=CommonTree;
}


tokens {
	JAVADOC;
	INLINE_TAG;
	DESCRIPTION;
	PARA_TAG;
	TEXT_LINE;
}


commentBody
	:	d=description paragraphTag* EOF
		-> ^(JAVADOC description paragraphTag*)
	;


description
	:	textLine*
		-> ^(DESCRIPTION textLine*)
	;


textLine
	:	textLineStart textLineContent* (NL | EOF!)
	|	NL
	;


textLineStart
	:	(LBRACE ATWORD)=> inlineTag
	|	WORD | STARS | WS | LBRACE | RBRACE | AT
	;


textLineContent
	:	(LBRACE ATWORD)=> inlineTag
	|	WORD | STARS | WS | LBRACE | RBRACE | AT | ATWORD
	;


inlineTag
	:	LBRACE ATWORD inlineTagContent* RBRACE
		-> ^(INLINE_TAG ATWORD inlineTagContent*)
	;


inlineTagContent
	:	WORD | STARS | WS | AT | NL
	;


paragraphTag
	:	ATWORD paragraphTagTail
		-> ^(PARA_TAG ATWORD paragraphTagTail)
	;


paragraphTagTail
	:	textLineContent* (NL textLine* | EOF)
		-> textLineContent* NL? textLine*
	;


STARS
    :		'*'+
    ;


LBRACE
    :		'{'
    ;


RBRACE
    :		'}'
    ;


AT
    :		'@'
    ;


WS
    :		(' ' | '\t')+
    ;


NL options {k=*;}
	:		('\r\n' | '\r' | '\n') WS? (STARS WS?)?
    ;


ATWORD
    :		'@' WORD WORD_TAIL
    ;


WORD
    :		~('\n' | ' ' | '\r' | '\t' | '{' | '}' | '@') WORD_TAIL
    ;


fragment WORD_TAIL
    :	(~('\n' | ' ' | '\r' | '\t' | '{' | '}'))*
    ;
