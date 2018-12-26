grammar t051treeRewriteASTo;
options {
    language=JavaScript;
    output=AST;
}
a : x=ID y=ID INT -> ^($x ^($y INT));
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\\n') {$channel=HIDDEN;} ;
