grammar t024finally;

options {
    language=JavaScript;
}

prog returns [events]
@init {events = [];}
@after {events.push('after');}
    :   ID {throw new Error("quux");}
    ;
    catch [e] {events.push('catch');}
    finally {events.push('finally');}

ID  :   ('a'..'z')+
    ;

WS  :   (' '|'\n'|'\r')+ {$channel=org.antlr.runtime.BaseRecognizer.HIDDEN}
    ;
