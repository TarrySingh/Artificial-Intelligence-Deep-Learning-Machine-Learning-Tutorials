grammar t051treeRewriteASTab;
options {
    language=JavaScript;
    output=AST;
}
a : ID INT -> ^(ID["root"] ^(ID INT)) | INT -> ^(ID["root"] INT) ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
