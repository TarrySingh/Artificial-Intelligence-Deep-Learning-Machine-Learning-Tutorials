grammar Expr;

options {
    language = Perl5;
}

@header {
}

@members {
    my %memory;
}

prog:   stat+ ;
                
stat:   expr NEWLINE { print "$expr.value\n"; }
    |   ID '=' expr NEWLINE
        { $memory{$ID.text} = $expr.value; }
    |   NEWLINE
    ;

expr returns [value]
    :   e=multExpr { $value = $e.value; }
        (   '+' e=multExpr { $value += $e.value; }
        |   '-' e=multExpr { $value -= $e.value; }
        )*
    ;

multExpr returns [value]
    :   e=atom { $value = $e.value; } ('*' e=atom { $value *= $e.value; })*
    ; 

atom returns [value]
    :   INT { $value = $INT.text; }
    |   ID
        {
            my $v = $memory{$ID.text};
            if (defined $v) {
                $value = $v;
            } else {
                print STDERR "undefined variable $ID.text\n";
            }
        }
    |   '(' expr ')' { $value = $expr.value; }
    ;

ID  :   ('a'..'z'|'A'..'Z')+ ;
INT :   '0'..'9'+ ;
NEWLINE:'\r'? '\n' ;
WS  :   (' '|'\t')+ { $self->skip(); } ;
