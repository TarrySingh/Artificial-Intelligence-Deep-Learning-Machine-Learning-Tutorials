package Point;

use ANTLR::Runtime::Class;

has 'x';
has 'y';

sub get_x {
    my ($self) = @_;
    return $self->x;
}

sub set_x {
    my ($self, $x) = @_;
    $self->x($x);
    return;
}

sub get_y {
    my ($self) = @_;
    return $self->y;
}

sub set_y {
    my ($self, $y) = @_;
    $self->y($y);
    return;
}

sub clear {
    my ($self) = @_;
    $self->x(0);
    $self->y(0);
}

package Point3D;
use ANTLR::Runtime::Class;

extends 'Point';

has 'z';

sub get_z {
    my ($self) = @_;
    return $self->z;
}

sub set_z {
    my ($self, $z) = @_;
    $self->z($z);
    return;
}

sub clear {
    my ($self) = @_;
    $self->SUPER::clear();
    $self->z(0);
}

package main;

use strict;
use warnings;

use Test::More;

plan tests => 12;

{
    my $p = Point->new();
    isa_ok($p, 'ANTLR::Runtime::Object');
    $p->set_x(5);
    $p->set_y(6);
    is($p->get_x(), 5);
    is($p->get_y(), 6);

    $p->clear();
    is($p->get_x(), 0);
    is($p->get_y(), 0);
}

{
    my $p = Point3D->new();
    isa_ok($p, 'Point');
    $p->set_x(5);
    $p->set_y(6);
    $p->set_z(7);
    is($p->get_x(), 5);
    is($p->get_y(), 6);
    is($p->get_z(), 7);

    $p->clear();
    is($p->get_x(), 0);
    is($p->get_y(), 0);
    is($p->get_z(), 0);
}
