lexer grammar IDLexer;
options { language = Perl5; }

ID  :   ('a'..'z'|'A'..'Z')+ ;
INT :   '0'..'9'+ ;
NEWLINE:'\r'? '\n' { $self->skip(); } ;
WS  :   (' '|'\t')+ { $channel = HIDDEN; } ;
