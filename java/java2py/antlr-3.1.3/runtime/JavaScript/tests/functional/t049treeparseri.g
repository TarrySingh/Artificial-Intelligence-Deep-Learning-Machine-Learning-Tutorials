grammar t049treeparseri;
options {
    language=JavaScript;
    output=AST;
}
a : x=ID INT? SEMI -> ^($x INT?) ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
SEMI : ';' ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
