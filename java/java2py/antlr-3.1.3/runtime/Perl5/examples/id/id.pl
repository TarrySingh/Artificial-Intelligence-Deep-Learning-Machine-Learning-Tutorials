#!/usr/bin/perl

use blib;

use English qw( -no_match_vars );
use ANTLR::Runtime::ANTLRStringStream;
use IDLexer;

use strict;
use warnings;

my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => "Hello World!\n42\n" });
my $lexer = IDLexer->new($input);

while (1) {
    my $token = $lexer->next_token();
    last if $token->get_type() == $IDLexer::EOF;

    print "text: ", $token->get_text(), "\n";
    print "type: ", $token->get_type(), "\n";
    print "pos: ", $token->get_line(), ':', $token->get_char_position_in_line(), "\n";
    print "channel: ", $token->get_channel(), "\n";
    print "token index: ", $token->get_token_index(), "\n";
    print "\n";
}
