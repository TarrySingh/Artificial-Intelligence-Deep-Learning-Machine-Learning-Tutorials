grammar t039labels;
options {
  language = Python;
}

a returns [l]
    : ids+=A ( ',' ids+=(A|B) )* C D w=. ids+=. F EOF
        { l = ($ids, $w) }
    ;

A: 'a'..'z';
B: '0'..'9';
C: a='A' { print $a };
D: a='FOOBAR' { print $a };
E: 'GNU' a=. { print $a };
F: 'BLARZ' a=EOF { print $a };

WS: ' '+  { $channel = HIDDEN };
