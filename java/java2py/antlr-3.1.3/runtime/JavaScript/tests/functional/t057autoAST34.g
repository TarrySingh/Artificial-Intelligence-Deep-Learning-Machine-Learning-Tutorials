grammar t057autoAST34;
options {language=JavaScript;output=AST;}
a returns [result] : id=ID id=ID {$result = "2nd id="+$id.text+";";} ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
