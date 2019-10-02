#!/usr/bin/perl

use strict;
use warnings;

use version;
use Carp;
use Digest;
use File::Spec;
use File::Spec::Unix;
use YAML::Tiny;

my $version = qv('0.0.1');

sub say {
    print @_, "\n";
}

my $basedir = '../..';

my $commands = {
    'help'   => \&help,
    'add'    => \&add,
    'status' => \&status,
};

my $help = {};

sub filetype {
    my ($path) = @_;

    if ($path =~ /\.(java|g)$/xms) {
        return 'text/plain';
    }
    else {
        return 'application/octet-stream';
    }
}

sub sha1sum {
    my ($filename) = @_;

    open my $in, '<', $filename or croak "Can't open $filename: $!";
    if (filetype($filename) =~ /^text\//xms) {
        # keep standard line feed conversion
    } else {
        if (!binmode $in) {
            croak "Can't binmode $filename: $!";
        }
    }
    my $sha1 = Digest->new('SHA-1');
    $sha1->addfile($in);
    my $digest = $sha1->hexdigest;
    close $in or warn "Can't close $filename: $!";
    return $digest;
}

my $inc_paths = [
    $basedir,
    "$basedir/runtime/Java/src",
];
    
sub resolve_file {
    my ($filename) = @_;

    my $resolved_file;
    if (-e $filename) {
        $resolved_file = $filename;
    }
    else {
        my @canidates 
            = grep { -e $_ } 
              map { File::Spec->catfile($_, $filename) } 
              @$inc_paths;
        $resolved_file = $canidates[0];
    }

    if (defined $resolved_file) {
        $resolved_file = File::Spec::Unix->canonpath($resolved_file);
    }

    return $resolved_file;
}

$help->{help} = << 'EOH';
help: Describe the usage of this program or its subcommands.
Usage: help [SUBCOMMAND...]
EOH

sub help {
    my ($cmd) = @_;

    if (defined $cmd) {
        print $help->{$cmd};
    }
    else {
        say << 'EOH';
Usage: port <subcommand> [options] [args]
EOH
        say "Available subcommands:";
        foreach my $cmd (keys %$help) {
            say "   $cmd";
        }
    }

}

$help->{add} = << 'EOH';
add: Adds the file to the list of ported files.
Usage: add PATH...
EOH

sub add {
    my ($filename) = @_;

    my $port = YAML::Tiny->read('port.yml');
    my $status = $port->[0]->{status};
    if (!defined $status) {
        $status = $port->[0]->{status} = {};
    }

    my $path = resolve_file($filename);
    if (!defined $path) {
        croak "File not found: $filename";
    }
    my $digest = sha1sum($path);
    $status->{$filename} = {
        'sha1' => $digest,
    };
    $port->write('port.yml');
}

$help->{status} = << 'EOH';
status: Print the status of the ported files.
usage: status [PATH...]
EOH

sub status {
    my $port = YAML::Tiny->read('port.yml');

    my $status = $port->[0]->{status};

    while (my ($filename, $fstatus) = each (%$status)) {
        my $path = resolve_file($filename);

        my $digest = sha1sum($path);

        if ($digest ne $fstatus->{sha1}) {
            say "M $filename";
        }
    }
}

my ($cmd, @args) = @ARGV;

if (defined $cmd) {
    my $cmd_f = $commands->{$cmd};
    if (defined $cmd_f) {
        $cmd_f->(@args);
    }
    else {
        say "Unknown command: '$cmd'";
        say "Type 'port help' for usage.";
        exit 1;
    }
}
else {
    say "Type 'port help' for usage.";
    exit 1;
}

__END__

=head1 NAME

port - ANTLR Perl 5 port status

=head1 VERSION

This documentation refers to port version 0.0.1

=head1 USAGE

    port help

    port status

=head1 DESCRIPTION

The primary language target for ANTLR is Java.  The Perl 5 port only follows
this primary target language.  This brings up the problem to follow the
changes made to the primary target, by knowing I<what> has changed and I<how>.

This tool keeps a database of file paths and content checksum.  Once the port
of a file (Java class, grammar, ...) is completed it is added to the
database (C<port add>).  This database can then be queried to check what
primary files have changed (C<port status>).  The revision control software
should be helpful to determine the actual changes.

=head1 AUTHOR

Ronald Blaschke (ron@rblasch.org)

