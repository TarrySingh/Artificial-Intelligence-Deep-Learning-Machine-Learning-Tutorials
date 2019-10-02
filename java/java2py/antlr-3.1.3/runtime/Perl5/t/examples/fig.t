use strict;
use warnings;

use lib qw( t/lib );

use Test::More;
use ANTLR::Runtime::Test;

plan tests => 1;

# The SimpleCalc grammar from the five minutes tutorial.
g_test_output_is({ grammar => <<'GRAMMAR', test_program => <<'CODE', expected => <<'OUTPUT' });
grammar Fig;
options { language = Perl5; }

@header {
use RunFig;
}

@members {
has 'instances' => (
    default => sub { {} }
);
}

file returns [objects]
    :   { $objects = []; }
        (object { push @$objects, $object.o; })+
    ;

object returns [o]
    :   qid v=ID?
        {
        $o = RunFig.newInstance($qid.text);
        if (defined $v) {
            $self->instances->{$v.text, $o);
        }
        }
        '{' assign[$o]* '}'
    ;

assign[o]
    :   ID '=' expr ';' {RunFig.setObjectProperty(o,$ID.text,$expr.value);}
    ;

expr returns [value]
    :   STRING  { $value = $STRING.text; }
    |   INT     { $value = Integer.valueOf($INT.text); }
    |   '$' ID  { $value = instances.get($ID.text); }
    |   '[' ']' { $value = new ArrayList(); }
    |   {ArrayList elements = new ArrayList(); }
        '[' e=expr { elements.add($e.value); }
            (',' e=expr { elements.add($e.value); })*
        ']'
        { $value = elements; }
    ;

qid :   ID ('.' ID)*
    ;

STRING : '"' .* '"' { setText(getText().substring(1, getText().length()-1)); } ;
INT :   '0'..'9'+ ;
ID  :   ('_'|'a'..'z'|'A'..'Z') ('_'|'a'..'z'|'A'..'Z'|'0'..'9')* ;
WS  :   (' '|'\n'|'\t')+ { $channel = $self->HIDDEN; } ;
CMT :   '/*' .* '*/'     { $channel = $self->HIDDEN; } ;
GRAMMAR

CODE

OUTPUT
