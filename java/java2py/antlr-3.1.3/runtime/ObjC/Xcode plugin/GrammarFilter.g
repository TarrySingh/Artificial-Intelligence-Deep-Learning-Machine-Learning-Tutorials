lexer grammar GrammarFilter;

options {
	language=ObjC;
	filter=true;
}

@ivars {
	id delegate;
}

@methodsdecl {
- (void) setDelegate:(id)theDelegate;
}

@methods {
- (void) setDelegate:(id)theDelegate
{
	delegate = theDelegate;	// not retained, will always be the object creating this lexer!
}
}

// figure out the grammar type in this file
GRAMMAR
	:	(	grammarType=GRAMMAR_TYPE {[delegate setGrammarType:$grammarType.text]; } WS
		|	/* empty, must be parser or combined grammar */ {[delegate setGrammarType:@"parser"]; [delegate setIsCombinedGrammar:NO]; }
		) 
		'grammar' WS grammarName=ID { [delegate setGrammarName:$grammarName.text]; } WS? ';'
	;

fragment
GRAMMAR_TYPE
	:  ('lexer'|'parser'|'tree')
	;
	
// find out if this grammar depends on other grammars
OPTIONS
	:	'options' WS? '{'
		( (WS? '//') => SL_COMMENT
		| (WS? '/*') => COMMENT
		| (WS? 'tokenVocab') => WS? 'tokenVocab' WS? '=' WS? tokenVocab=ID WS? ';' { [delegate setDependsOnVocab:$tokenVocab.text]; }
		| WS? ID WS? '=' WS? ID WS?';'
		)*
		WS? '}'
	;

// look for lexer rules when in parser grammar -> this 
LEXER_RULE
	:	('A'..'Z') ID? WS? ':' (options {greedy=false;} : .)* ';' { [delegate setIsCombinedGrammar:YES]; }
	;

// ignore stuff in comments
COMMENT
    :   '/*' (options {greedy=false;} : . )* '*/'
    ;

SL_COMMENT
    :   '//' (options {greedy=false;} : . )* '\n'
    ;

// throw away all actions, as they might confuse LEXER_RULE
ACTION
	:	'{' (options {greedy=false;} : .)* '}'
	;
// similarly throw away strings
STRING
	:	'\'' (options {greedy=false;} : .)* '\''
	;

fragment
ID  :   ('a'..'z'|'A'..'Z'|'_'|'0'..'9')+
    ;
	
WS  :   (' '|'\t'|'\n')+
    ;