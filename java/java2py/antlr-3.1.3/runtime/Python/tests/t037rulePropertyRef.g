grammar t037rulePropertyRef;
options {
  language = Python;
}

a returns [bla]
@after {
    $bla = $start, $stop, $text
}
    : A+
    ;

A: 'a'..'z';

WS: ' '+  { $channel = HIDDEN };
