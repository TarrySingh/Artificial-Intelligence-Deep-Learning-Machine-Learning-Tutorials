use strict;
use warnings;

use Test::More;

plan tests => 7;

use ANTLR::Runtime::ANTLRStringStream;

{
    my $s = ANTLR::Runtime::ANTLRStringStream->new({ input => 'ABC' });
    is ($s->LA(1), 'A');
    $s->consume();
    is ($s->LA(1), 'B');
}

{
    my $s = ANTLR::Runtime::ANTLRStringStream->new({ input => 'ABC' });
    is($s->LA(0), undef);
    is($s->LA(1), 'A');
    is($s->LA(2), 'B');
    is($s->LA(3), 'C');
    is($s->LA(4), ANTLR::Runtime::ANTLRStringStream->EOF);
}
