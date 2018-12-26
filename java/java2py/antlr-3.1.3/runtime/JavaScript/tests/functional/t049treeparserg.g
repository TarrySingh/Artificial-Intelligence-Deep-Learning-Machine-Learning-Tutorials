grammar t049treeparserg;
options {
    language=JavaScript;
    output=AST;
}
a : ID INT? SEMI -> ^(ID INT?) SEMI ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
SEMI : ';' ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
