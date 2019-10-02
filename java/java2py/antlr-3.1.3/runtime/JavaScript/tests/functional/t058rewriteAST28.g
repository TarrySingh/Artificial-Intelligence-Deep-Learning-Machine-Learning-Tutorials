
grammar t058rewriteAST28;
options {language=JavaScript;output=AST;}
a : 'var' (ID ':' type ';')+ -> ^('var' ^(':' ID type)+) ;
type : 'int' | 'float' ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;