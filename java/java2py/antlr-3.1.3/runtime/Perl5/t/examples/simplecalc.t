use strict;
use warnings;

use lib qw( t/lib );

use Test::More;
use ANTLR::Runtime::Test;

plan tests => 1;

# The SimpleCalc grammar from the five minutes tutorial.
g_test_output_is({ grammar => <<'GRAMMAR', test_program => <<'CODE', expected => <<'OUTPUT' });
grammar SimpleCalc;
options { language = Perl5; }

tokens {
	PLUS 	= '+' ;
	MINUS	= '-' ;
	MULT	= '*' ;
	DIV	= '/' ;
}

/*------------------------------------------------------------------
 * PARSER RULES
 *------------------------------------------------------------------*/

expr	: term ( ( PLUS | MINUS )  term )* ;

term	: factor ( ( MULT | DIV ) factor )* ;

factor	: NUMBER ;

/*------------------------------------------------------------------
 * LEXER RULES
 *------------------------------------------------------------------*/

NUMBER	: (DIGIT)+ ;

WHITESPACE : ( '\t' | ' ' | '\r' | '\n'| '\u000C' )+ 	{ $channel = HIDDEN; } ;

fragment DIGIT	: '0'..'9' ;
GRAMMAR
use strict;
use warnings;

use ANTLR::Runtime::ANTLRStringStream;
use ANTLR::Runtime::CommonTokenStream;
use ANTLR::Runtime::RecognitionException;
use SimpleCalcLexer;
use SimpleCalcParser;

my @examples = (
    '1',
    '1 + 1',
    '1 +',
    '1 * 2 + 3',
);

foreach my $example (@examples) {
    my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => $example });
    my $lexer = SimpleCalcLexer->new({ input => $input });
    my $tokens = ANTLR::Runtime::CommonTokenStream->new({ token_source => $lexer });
    my $parser = SimpleCalcParser->new({ input => $tokens });
    eval {
        $parser->expr();
        if ($parser->get_number_of_syntax_errors() == 0) {
            print "$example: good\n";
        }
        else {
            print "$example: bad\n";
        }
    };
    if (my $ex = ANTLR::Runtime::RecognitionException->caught()) {
        print "$example: error\n";
    } elsif ($ex = Exception::Class->caught()) {
        print "$example: error: $ex\n";
        ref $ex ? $ex->rethrow() : die $ex;
    }
}
CODE
1: good
1 + 1: good
1 +: bad
1 * 2 + 3: good
OUTPUT

__END__
