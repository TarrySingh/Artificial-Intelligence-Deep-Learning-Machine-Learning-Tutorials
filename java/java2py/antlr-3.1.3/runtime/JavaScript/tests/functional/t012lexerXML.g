/** XML parser by Oliver Zeigermann October 10, 2005 */
lexer grammar t012lexerXML;
options {
  language = JavaScript;
}

@lexer::members {
this.lout = [];
this.output = function(line) {
    this.lout.push(line);
};
}

DOCUMENT
    :  XMLDECL? WS? DOCTYPE? WS? ELEMENT WS? 
    ;

fragment DOCTYPE
    :
        '<!DOCTYPE' WS rootElementName=GENERIC_ID 
        {this.output("ROOTELEMENT: "+$rootElementName.text)}
        WS
        ( 
            ( 'SYSTEM' WS sys1=VALUE
                {this.output("SYSTEM: "+$sys1.text)}
                
            | 'PUBLIC' WS pub=VALUE WS sys2=VALUE
                {this.output("PUBLIC: "+$pub.text)}
                {this.output("SYSTEM: "+$sys2.text)}   
            )
            ( WS )?
        )?
        ( dtd=INTERNAL_DTD
            {this.output("INTERNAL DTD: "+$dtd.text)}
        )?
		'>'
	;

fragment INTERNAL_DTD : '[' (options {greedy=false;} : .)* ']' ;

fragment PI :
        '<?' target=GENERIC_ID WS? 
          {this.output("PI: "+$target.text)}
        ( ATTRIBUTE WS? )*  '?>'
	;

fragment XMLDECL :
        '<?' ('x'|'X') ('m'|'M') ('l'|'L') WS? 
          {this.output("XML declaration")}
        ( ATTRIBUTE WS? )*  '?>'
	;


fragment ELEMENT
    : ( START_TAG
            (ELEMENT
            | t=PCDATA
                {this.output("PCDATA: \""+$t.text+"\"")}
            | t=CDATA
                {this.output("CDATA: \""+$t.text+"\"")}
            | t=COMMENT
                {this.output("Comment: \""+$t.text+"\"")}
            | pi=PI
            )*
            END_TAG
        | EMPTY_ELEMENT
        )
    ;

fragment START_TAG 
    : '<' WS? name=GENERIC_ID WS?
          {this.output("Start Tag: "+$name.text)}
        ( ATTRIBUTE WS? )* '>'
    ;

fragment EMPTY_ELEMENT 
    : '<' WS? name=GENERIC_ID WS?
          {this.output("Empty Element: "+$name.text)}
        ( ATTRIBUTE WS? )* '/>'
    ;

fragment ATTRIBUTE 
    : name=GENERIC_ID WS? '=' WS? value=VALUE
        {this.output("Attr: "+$name.text+"="+$value.text)}
    ;

fragment END_TAG 
    : '</' WS? name=GENERIC_ID WS? '>'
        {this.output("End Tag: "+$name.text)}
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

