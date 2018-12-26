use strict;
use warnings;

use File::Spec;
use Java::JVM::Classfile;

use Test::More tests => 29;

sub class_name_to_java {
    my ($name) = @_;

    my $tmp = $name;
    $tmp =~ s/ANTLR::Runtime/org.antlr.runtime/;
    $tmp =~ s/::/./g;

    return $tmp;
}

sub java_class_name_to_perl {
    my ($name) = @_;

    my $tmp = $name;
    $tmp =~ s/org\.antlr\.runtime/ANTLR::Runtime/;
    $tmp =~ s/\./::/g;

    return $tmp;
}

sub resolve_java_class_file {
    my ($name, $basedir) = @_;

    my $tmp = $name;
    $tmp =~ s!\.!/!g;
    $tmp .= '.class';

    return File::Spec->catfile($basedir, $tmp);
}

sub java_constant_name_to_perl {
}

sub java_method_name_to_perl {
    my ($name) = @_;

    if ($name eq '<init>') {
        return 'new';
    }
    # add special cases here
    else {
        my $tmp = $name;
        $tmp =~ s/([a-z])([A-Z])/$1_\L$2\E/g;

        return $tmp;
    }
}

my @java_class_names = qw(
    org.antlr.runtime.BitSet
);

foreach my $java_class_name (@java_class_names) {
    my $java_class_file = resolve_java_class_file($java_class_name,
        '../../build/rtclasses');

    my $java_class;
    {
        local $SIG{'__WARN__'} = sub {};
        $java_class = Java::JVM::Classfile->new($java_class_file);
    }

    my $class_name = java_class_name_to_perl($java_class_name);
    use_ok($class_name);
    print map { "$_\n" } ANTLR::Runtime::BitSet->can();
    print "---\n";

    eval { $class_name->new() };
    print join "\n", ANTLR::Runtime::BitSet->can();
    print "\n";

    my $java_fields = $java_class->fields;
    foreach my $java_field (@$java_fields) {
        next if grep { $_ eq 'private' } @{$java_field->access_flags};

        my $field_name = $java_field->name;
        ok($class_name->can($field_name), $field_name);
    }

    my $java_methods = $java_class->methods;
    foreach my $java_method (@$java_methods) {
        next if grep { $_ eq 'private' } @{$java_method->access_flags};

        my $method_name = java_method_name_to_perl($java_method->name);
        ok($class_name->can($method_name), $method_name);
    }
}
