grammar t035ruleLabelPropertyRef;
options {
  language = JavaScript;
}

a returns [bla]: t=b
        {
            $bla = [$t.start, $t.stop, $t.text];
        }
    ;

b: A+;

A: 'a'..'z';

WS: ' '+  { $channel = HIDDEN; };
