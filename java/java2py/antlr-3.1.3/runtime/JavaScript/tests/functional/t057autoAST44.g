grammar t057autoAST44;
options {language=JavaScript;output=AST;}
a returns [result] : ID b {
/* @todo */
/* $result = $b.i.toString() + '\n'; */
} ;
b returns [i] : INT {$i=parseInt($INT.text);} ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
