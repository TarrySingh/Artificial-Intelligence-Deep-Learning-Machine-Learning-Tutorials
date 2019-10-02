use strict;
use warnings;

use Test::More;

eval "use Test::Pod::Coverage";
if ($@) {
    plan skip_all => "Test::Pod::Coverage required for testing POD coverage: $@";
}
all_pod_coverage_ok();
