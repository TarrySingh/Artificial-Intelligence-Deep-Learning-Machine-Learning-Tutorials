grammar t051treeRewriteASTr;
options {
    language=JavaScript;
    output=AST;
}
a : x=ID INT -> ^($x INT);
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
