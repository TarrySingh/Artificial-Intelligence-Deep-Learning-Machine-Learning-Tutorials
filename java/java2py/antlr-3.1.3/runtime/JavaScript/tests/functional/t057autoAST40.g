grammar t057autoAST40;
options {language=JavaScript;output=AST;}
a returns [result]: x+=b x+=b {
t=$x[1]
$result = "2nd x="+t.toStringTree()+',';
};
b : ID;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
