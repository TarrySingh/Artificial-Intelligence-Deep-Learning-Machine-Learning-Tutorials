use strict;
use warnings;

use ExtUtils::MakeMaker;
use Test::More tests => 1;

my $file = 'lib/ANTLR/Runtime.pm';

my $version = MM->parse_version($file);

# classic CPAN
#like($version, qr/^\d+\.\d{2,}(_\d{2,})?$/);

# version.pm
like($version, qr/^\d+\.\d+\.\d+(?:_\d+)?$/);
