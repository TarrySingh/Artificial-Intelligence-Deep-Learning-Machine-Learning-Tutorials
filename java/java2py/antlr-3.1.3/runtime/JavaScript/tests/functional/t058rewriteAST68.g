
grammar t058rewriteAST68;
options {language=JavaScript;output=AST;} 
a : x=(INT|ID) -> $x ;
INT: '0'..'9'+;
ID : 'a'..'z'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;