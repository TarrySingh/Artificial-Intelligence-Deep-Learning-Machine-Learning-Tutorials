#!perl

use strict;
use warnings;

use ANTLR::Runtime::ANTLRFileStream;
use ANTLR::Runtime::CommonTokenStream;
use ANTLR::Runtime::RecognitionException;
use SimpleCalcLexer;
use SimpleCalcParser;

my $input = ANTLR::Runtime::ANTLRFileStream->new({ file_name => $ARGV[0] });
my $lexer = SimpleCalcLexer->new({ input => $input });
my $tokens = ANTLR::Runtime::CommonTokenStream->new({ token_source => $lexer });
my $parser = SimpleCalcParser->new({ input => $tokens });
eval {
    $parser->expr();
    print "ok\n";
    print "errors: ", $parser->get_number_of_syntax_errors(), "\n";
    print "failed? ", $parser->failed(), "\n";
};
if (my $ex = ANTLR::Runtime::RecognitionException->caught()) {
    print $ex->trace, "\n";
}
elsif ($ex = $@) {
    die $ex;
}
