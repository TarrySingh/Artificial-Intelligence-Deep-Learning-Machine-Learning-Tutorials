grammar t057autoAST41;
options {language=JavaScript;output=AST;}
a returns [result] : ( x+=b^ )+ {
$result = "x="+$x[1].toStringTree()+',';
} ;
b : ID;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
