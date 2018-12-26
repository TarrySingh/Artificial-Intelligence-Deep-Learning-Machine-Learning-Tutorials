grammar t036multipleReturnValues;
options {
  language = Python;
}

a returns [foo, bar]: A
        {
            $foo = "foo";
            $bar = "bar";
        }
    ;

A: 'a'..'z';

WS  :
        (   ' '
        |   '\t'
        |  ( '\n'
            |	'\r\n'
            |	'\r'
            )
        )+
        { $channel = HIDDEN }
    ;    

