grammar t057autoAST42;
options {language=JavaScript;output=AST;}
a returns [result] : x+=b! x+=b {
$result = "1st x="+$x[0].toStringTree()+',';
} ;
b : ID;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
