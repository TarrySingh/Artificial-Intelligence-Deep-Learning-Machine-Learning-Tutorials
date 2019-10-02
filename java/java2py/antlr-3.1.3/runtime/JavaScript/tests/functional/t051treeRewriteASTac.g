grammar t051treeRewriteASTac;
options {
    language=JavaScript;
    output=AST;
}
a : ID INT -> ^(ID["root"] INT);
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
