grammar t051treeRewriteASTu;
options {
    language=JavaScript;
    output=AST;
}
a : ID INT -> ID INT | INT ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
