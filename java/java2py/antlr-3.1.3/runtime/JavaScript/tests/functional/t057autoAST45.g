grammar t057autoAST45;
options { language=JavaScript;output=AST; }
r : (INT|ID)+ ; 
ID : 'a'..'z' + ;
INT : '0'..'9' +;
WS: (' ' | '\n' | '\\t')+ {$channel = HIDDEN;};
