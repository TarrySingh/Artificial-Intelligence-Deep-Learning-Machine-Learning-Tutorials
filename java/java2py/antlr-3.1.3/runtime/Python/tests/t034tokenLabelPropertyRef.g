grammar t034tokenLabelPropertyRef;
options {
  language = Python;
}

a: t=A
        {
            print $t.text
            print $t.type
            print $t.line
            print $t.pos
            print $t.channel
            print $t.index
            #print $t.tree
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

