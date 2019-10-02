grammar t049treeparserh;
options {
    language=JavaScript;
    output=AST;
}
a : x=ID INT? (y=ID)? SEMI -> ^($x INT? $y?) SEMI ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
SEMI : ';' ;
WS : (' '|'\\n') {$channel=HIDDEN;} ;
