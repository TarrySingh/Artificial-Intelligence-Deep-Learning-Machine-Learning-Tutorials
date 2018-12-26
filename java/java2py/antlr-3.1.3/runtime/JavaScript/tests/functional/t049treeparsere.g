grammar t049treeparsere;
options {
    language=JavaScript;
    output=AST;
}
a : ID INT+ PERIOD;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
SEMI : ';' ;
PERIOD : '.' ;
WS : (' '|'\n') {$channel=HIDDEN;} ;
