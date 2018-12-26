use strict;
use warnings;

use File::Spec;
use English qw(-no_match_vars);

use Test::More;

eval {
    require Test::Perl::Critic;
};
if ( $EVAL_ERROR ) {
   my $msg = 'Test::Perl::Critic required to criticise code';
   plan( skip_all => $msg );
}

my $rcfile = File::Spec->catfile( 't', 'author', 'perlcriticrc' );
Test::Perl::Critic->import( -profile => $rcfile );
all_critic_ok();
