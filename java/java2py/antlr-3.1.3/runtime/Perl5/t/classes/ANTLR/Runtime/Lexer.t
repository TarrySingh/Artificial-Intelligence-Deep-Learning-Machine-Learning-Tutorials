use strict;
use warnings;

use Test::More;

plan tests => 1;

use ANTLR::Runtime::ANTLRStringStream;
use ANTLR::Runtime::Lexer;

{
    my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => 'ABC' });
    my $lexer = ANTLR::Runtime::Lexer->new({ input => $input });
    ok(defined $lexer);
}
