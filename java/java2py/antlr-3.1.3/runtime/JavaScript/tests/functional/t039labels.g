grammar t039labels;
options {
  language = JavaScript;
}

a returns [l]
    : ids+=A ( ',' ids+=(A|B) )* C D w=. ids+=. F EOF
        { l = [$ids, $w]; }
    ;

A: 'a'..'z';
B: '0'..'9';
C: a='A';
D: a='FOOBAR';
E: 'GNU' a=.;
F: 'BLARZ' a=EOF;

WS: ' '+  { $channel = HIDDEN };
