#!/usr/bin/perl

use strict;
use warnings;

use blib;

use ANTLR::Runtime::ANTLRStringStream;
use ANTLR::Runtime::CommonTokenStream;
use MExprLexer;
use MExprParser;

while (<>) {
    my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => $_ });
    my $lexer = MExprLexer->new($input);

    my $tokens = ANTLR::Runtime::CommonTokenStream->new({ token_source => $lexer });
    my $parser = MExprParser->new($tokens);
    $parser->prog();
}
