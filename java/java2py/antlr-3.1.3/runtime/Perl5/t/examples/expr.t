use strict;
use warnings;

use lib qw( t/lib );

use Test::More;
use ANTLR::Runtime::Test;

plan tests => 1;

# The SimpleCalc grammar from the five minutes tutorial.
g_test_output_is({ grammar => <<'GRAMMAR', test_program => <<'CODE', expected => <<'OUTPUT' });
grammar Expr;
options { language = Perl5; }
@header {}

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
GRAMMAR
use strict;
use warnings;

use ANTLR::Runtime::ANTLRStringStream;
use ANTLR::Runtime::CommonTokenStream;
use ExprLexer;
use ExprParser;

my $in = << 'EOT';
1 + 1
8 - 1
a = 10
b = 13
2 * a + b + 1
EOT

my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => $in });
my $lexer = ExprLexer->new({ input => $input });

my $tokens = ANTLR::Runtime::CommonTokenStream->new({ token_source => $lexer });
my $parser = ExprParser->new({ input => $tokens });
$parser->prog();
CODE
2
7
34
OUTPUT
