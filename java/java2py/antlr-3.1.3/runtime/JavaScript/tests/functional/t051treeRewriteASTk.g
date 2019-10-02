grammar t051treeRewriteASTk;
options {
    language=JavaScript;
    output=AST;
}
a : ID INT -> ^(ID INT);
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
