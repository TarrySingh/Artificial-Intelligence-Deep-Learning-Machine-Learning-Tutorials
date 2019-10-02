grammar t041parameters;
options {
  language = JavaScript;
}

a[arg1, arg2] returns [l]
    : A+ EOF
        { 
            l = [$arg1, $arg2];
            $arg1 = "gnarz";
        }
    ;

A: 'a'..'z';

WS: ' '+  { $channel = HIDDEN; };
