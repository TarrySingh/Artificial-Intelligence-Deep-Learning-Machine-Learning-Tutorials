grammar MExpr;

options {
  language = Perl5;
}

prog:   stat+ ;
                
stat:   expr NEWLINE { print "$expr.value\n"; }
    |   NEWLINE
    ;

expr returns [value]
    :   e=atom { $value = $e.value; }
        (   '+' e=atom { $value += $e.value; }
        |   '-' e=atom { $value -= $e.value; }
        )*
    ;

atom returns [value]
    :   INT { $value = $INT.text; }
    |   '(' expr ')' { $value = $expr.value; }
    ;

ID  :   ('a'..'z'|'A'..'Z')+ ;
INT :   '0'..'9'+ ;
NEWLINE:'\r'? '\n' ;
WS  :   (' '|'\t')+ { $self->skip(); } ;
