use strict;
use warnings;

use Test::More;

eval "use Test::Pod";
if ($@) {
    plan skip_all => "Test::Pod required for testing POD: $@";
}
all_pod_files_ok();
