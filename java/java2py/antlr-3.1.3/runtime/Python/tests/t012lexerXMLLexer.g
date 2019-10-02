lexer grammar t012lexerXMLLexer;
options {
  language = Python;
}

@header {
from cStringIO import StringIO
}

@lexer::init {
self.outbuf = StringIO()
}

@lexer::members {
def output(self, line):
    self.outbuf.write(line.encode('utf-8') + "\n")
}

DOCUMENT
    :  XMLDECL? WS? DOCTYPE? WS? ELEMENT WS? 
    ;

fragment DOCTYPE
    :
        '<!DOCTYPE' WS rootElementName=GENERIC_ID 
        {self.output("ROOTELEMENT: "+rootElementName.text)}
        WS
        ( 
            ( 'SYSTEM' WS sys1=VALUE
                {self.output("SYSTEM: "+sys1.text)}
                
            | 'PUBLIC' WS pub=VALUE WS sys2=VALUE
                {self.output("PUBLIC: "+pub.text)}
                {self.output("SYSTEM: "+sys2.text)}   
            )
            ( WS )?
        )?
        ( dtd=INTERNAL_DTD
            {self.output("INTERNAL DTD: "+dtd.text)}
        )?
		'>'
	;

fragment INTERNAL_DTD : '[' (options {greedy=false;} : .)* ']' ;

fragment PI :
        '<?' target=GENERIC_ID WS? 
          {self.output("PI: "+target.text)}
        ( ATTRIBUTE WS? )*  '?>'
	;

fragment XMLDECL :
        '<?' ('x'|'X') ('m'|'M') ('l'|'L') WS? 
          {self.output("XML declaration")}
        ( ATTRIBUTE WS? )*  '?>'
	;


fragment ELEMENT
    : ( START_TAG
            (ELEMENT
            | t=PCDATA
                {self.output("PCDATA: \""+$t.text+"\"")}
            | t=CDATA
                {self.output("CDATA: \""+$t.text+"\"")}
            | t=COMMENT
                {self.output("Comment: \""+$t.text+"\"")}
            | pi=PI
            )*
            END_TAG
        | EMPTY_ELEMENT
        )
    ;

fragment START_TAG 
    : '<' WS? name=GENERIC_ID WS?
          {self.output("Start Tag: "+name.text)}
        ( ATTRIBUTE WS? )* '>'
    ;

fragment EMPTY_ELEMENT 
    : '<' WS? name=GENERIC_ID WS?
          {self.output("Empty Element: "+name.text)}
        ( ATTRIBUTE WS? )* '/>'
    ;

fragment ATTRIBUTE 
    : name=GENERIC_ID WS? '=' WS? value=VALUE
        {self.output("Attr: "+name.text+"="+value.text)}
    ;

fragment END_TAG 
    : '</' WS? name=GENERIC_ID WS? '>'
        {self.output("End Tag: "+name.text)}
    ;

fragment COMMENT
	:	'<!--' (options {greedy=false;} : .)* '-->'
	;

fragment CDATA
	:	'<![CDATA[' (options {greedy=false;} : .)* ']]>'
	;

fragment PCDATA : (~'<')+ ; 

fragment VALUE : 
        ( '\"' (~'\"')* '\"'
        | '\'' (~'\'')* '\''
        )
	;

fragment GENERIC_ID 
    : ( LETTER | '_' | ':') 
        ( options {greedy=true;} : LETTER | '0'..'9' | '.' | '-' | '_' | ':' )*
	;

fragment LETTER
	: 'a'..'z' 
	| 'A'..'Z'
	;

fragment WS  :
        (   ' '
        |   '\t'
        |  ( '\n'
            |	'\r\n'
            |	'\r'
            )
        )+
    ;    

