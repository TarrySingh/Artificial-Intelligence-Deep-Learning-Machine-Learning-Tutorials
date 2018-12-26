grammar t051treeRewriteASTz;
options {
    language=JavaScript;
    output=AST;
}
a : 'boo' ID INT -> ^('boo' ^(ID INT)) ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
