#!/usr/bin/perl

use blib;

use English qw( -no_match_vars );
use ANTLR::Runtime::ANTLRStringStream;
use TLexer;

use strict;
use warnings;

my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => '010' });
my $lexer = TLexer->new($input);

while (1) {
    my $token = $lexer->next_token();
    last if $token->get_type() == $TLexer::EOF;

    print "type: ", $token->get_type(), "\n";
    print "text: ", $token->get_text(), "\n";
    print "\n";
}
