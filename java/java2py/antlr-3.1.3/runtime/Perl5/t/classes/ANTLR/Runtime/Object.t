use strict;
use warnings;

use Test::More;

plan tests => 8;

use ANTLR::Runtime::Object;
use Params::Validate qw( :types );


package ANTLR::Runtime::TestObject;
use ANTLR::Runtime::Class;

use ANTLR::Runtime::Object::Signature;

sub add :method :Signature(int $lhs, int $rhs  --> int) {
    #print "lhs = $lhs, rhs = $rhs\n";
    return $lhs + $rhs;
}

package main;

{
    my ($obj, $a) = ANTLR::Runtime::Object->unpack_method_args(
        [ undef, 1 ],
        {
            spec => [
                {
                    name => 'a',
                },
            ]
        }
    );

    is($obj, undef);
    is($a, 1);
}

{
    my ($obj, $a) = ANTLR::Runtime::Object->unpack_method_args(
        [ undef, { a => 1 } ],
        {
            spec => [
                {
                    name => 'a',
                },
            ]
        }
    );

    is ($obj, undef);
    is ($a, 1);
}

{
    # We might expect the following to unpack $b == 2,
    # but this might be too fragile?  Until specified,
    # it's (more or less) by accident treated as positional.
    my ($obj, $a, $b) = ANTLR::Runtime::Object->unpack_method_args(
        [ undef, 1, { b => 2 }],
        {
            spec => [
                {
                    name => 'a',
                },
                {
                    name => 'b',
                },
            ]
        }
    );

    is($obj, undef);
    is($a, 1);
    is_deeply($b, { b=> 2 });
}

{
    is(ANTLR::Runtime::TestObject->add(1, 1), 2);
}

1;
