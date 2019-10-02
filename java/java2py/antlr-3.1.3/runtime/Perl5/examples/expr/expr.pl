#!/usr/bin/perl

use strict;
use warnings;

use blib;

use ANTLR::Runtime::ANTLRStringStream;
use ANTLR::Runtime::CommonTokenStream;
use ExprLexer;
use ExprParser;

my $in;
{
    undef $/;
    $in = <>;
}

my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => $in });
my $lexer = ExprLexer->new($input);

my $tokens = ANTLR::Runtime::CommonTokenStream->new({ token_source => $lexer });
my $parser = ExprParser->new({ input => $tokens });
$parser->prog();
