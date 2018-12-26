grammar t024finally;

options {
    language=Python;
}

prog returns [events]
@init {events = []}
@after {events.append('after')}
    :   ID {raise RuntimeError}
    ;
    catch [RuntimeError] {events.append('catch')}
    finally {events.append('finally')}

ID  :   ('a'..'z')+
    ;

WS  :   (' '|'\n'|'\r')+ {$channel=HIDDEN}
    ;
