use strict;
use warnings;

use Test::More;

plan tests => 3;

use ANTLR::Runtime::Exception;

{
    # pick any error
    $! = 1;
    my $expected = "$!";
    my $ex = ANTLR::Runtime::Exception->new();
    is $ex->message, $expected;
}

{
    my $ex = ANTLR::Runtime::Exception->new({ message => 'test error message' });
    is $ex->message, 'test error message';
}

{
    eval {
        ANTLR::Runtime::Exception->throw(message => 'test error message');
    };
    my $ex = ANTLR::Runtime::Exception->caught();
    is $ex->message, 'test error message';
}
