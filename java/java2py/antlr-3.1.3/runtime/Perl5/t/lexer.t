use strict;
use warnings;

use FindBin;
use lib qw( t/lib );

use File::Slurp;

use Test::More;
use ANTLR::Runtime::Test;

plan tests => 2;

sub grammar_file {
    my ($file) = @_;
    return read_file("t/$file");
}

# A simple test: try to lex one possible token.
g_test_output_is({ grammar => <<'GRAMMAR', test_program => <<'CODE', expected => <<'OUTPUT' });
/* This is a comment.  Note that we're in the ANTLR grammar here, so it's not
   a Perl '#' comment, and may be multi line... */
// ... or a single line comment
lexer grammar INTLexer;
/* Set target language to Perl5. */
options { language = Perl5; }

/* Lexer rule for an integer. */
INT : '0'..'9'+;
GRAMMAR
use strict;
use warnings;

use ANTLR::Runtime::ANTLRStringStream;
use INTLexer;

my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => '123' });
my $lexer = INTLexer->new({ input => $input });
while ((my $_ = $lexer->next_token())) {
    print $_->get_text(), "\n";
}
CODE
123
OUTPUT

# Multiple choice, including 'skip' and 'hide' actions.
g_test_output_is({ grammar => <<'GRAMMAR', test_program => <<'CODE', expected => <<'OUTPUT' });
lexer grammar IDLexer;
options { language = Perl5; }

ID      : ('a'..'z'|'A'..'Z')+ ;
INT     : '0'..'9'+ ;
NEWLINE : '\r'? '\n'  { $self->skip() } ;
WS      : (' '|'\t')+ { $channel = HIDDEN } ;
GRAMMAR
use strict;
use warnings;

use ANTLR::Runtime::ANTLRStringStream;
use IDLexer;

my $input = ANTLR::Runtime::ANTLRStringStream->new({ input => "Hello World!\n42\n" });
my $lexer = IDLexer->new({ input => $input });

while (1) {
    my $token = $lexer->next_token();
    last if $token->get_type() == IDLexer->EOF;

    print "text: '", $token->get_text(), "'\n";
    print "type: ",  $token->get_type(), "\n";
    print "pos: ",   $token->get_line(), ':', $token->get_char_position_in_line(), "\n";
    print "channel: ",     $token->get_channel(), "\n";
    print "token index: ", $token->get_token_index(), "\n";
    print "\n";
}
CODE
text: 'Hello'
type: 4
pos: 1:0
channel: 0
token index: -1

text: ' '
type: 7
pos: 1:5
channel: 99
token index: -1

text: 'World'
type: 4
pos: 1:6
channel: 0
token index: -1

text: '42'
type: 5
pos: 2:0
channel: 0
token index: -1

OUTPUT

=begin SKIP doesn't compile yet

g_test_output_is({ grammar => scalar grammar_file('XMLLexer.g'), test_program => <<'CODE', expected => <<'OUTPUT' });
use English qw( -no_match_vars );
use ANTLR::Runtime::ANTLRStringStream;
use XMLLexer;

use strict;
use warnings;

my $input = ANTLR::Runtime::ANTLRStringStream->new(<< 'XML');
<?xml version='1.0'?>
<test>foo</test>
XML
my $lexer = IDLexer->new($input);
while ((my $_ = $lexer->next_token())) {
}
CODE
XML declaration
PCDATA: "foo"
OUTPUT
}

=end SKIP
