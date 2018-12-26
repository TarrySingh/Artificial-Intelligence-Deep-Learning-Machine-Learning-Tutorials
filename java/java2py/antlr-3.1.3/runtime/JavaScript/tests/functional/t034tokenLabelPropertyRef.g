grammar t034tokenLabelPropertyRef;
options {
  language = JavaScript;
}

@header {
var xlog = [];
}

a: t=A
        {
            xlog.push($t.text);
            xlog.push($t.type);
            xlog.push($t.line);
            xlog.push($t.pos);
            xlog.push($t.channel);
            xlog.push($t.index);
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
        { $channel = org.antlr.runtime.Token.HIDDEN_CHANNEL; }
    ;    

