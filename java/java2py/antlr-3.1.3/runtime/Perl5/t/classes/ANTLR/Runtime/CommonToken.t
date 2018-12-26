use strict;
use warnings;

use Test::More;

plan tests => 4;

use ANTLR::Runtime::CommonToken;

{
    my $token = ANTLR::Runtime::CommonToken->new({
        input => undef,
        type => 0,
        channel => 0,
        start => 0,
        stop => 1,
    });
    is($token->get_start_index(), 0);
}

ok(ANTLR::Runtime::Token->EOF_TOKEN == ANTLR::Runtime::Token->EOF_TOKEN);
ok(!(ANTLR::Runtime::Token->EOF_TOKEN != ANTLR::Runtime::Token->EOF_TOKEN));
ok(!ANTLR::Runtime::Token->EOF_TOKEN);
