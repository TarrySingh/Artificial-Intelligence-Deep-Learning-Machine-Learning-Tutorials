grammar t053heteroT16;
options {
    language=JavaScript;
    output=AST;
}
a : ID ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
