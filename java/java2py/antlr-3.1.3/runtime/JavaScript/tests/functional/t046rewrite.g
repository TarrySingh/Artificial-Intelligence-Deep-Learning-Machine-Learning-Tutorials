grammar t046rewrite;
options {
    language=JavaScript;
}

program
@init {
    var start = this.input.LT(1);
}
    :   method+
        {
        this.input.insertBefore(start,"public class Wrapper {\n")
        this.input.insertAfter($method.stop, "\n}\n")
        }
    ;

method
    :   m='method' ID '(' ')' body
        {this.input.replace($m, "public void");}
    ; 

body
scope {
    decls
}
@init {
    $body::decls = {};
}
    :   lcurly='{' stat* '}'
        {
        var it;
        for (it in $body::decls) {
            this.input.insertAfter($lcurly, "\nint "+it+";")
        }
        }
    ;

stat:   ID '=' expr ';' {$body::decls[$ID.text] = true;}
    ;

expr:   mul ('+' mul)* 
    ;

mul :   atom ('*' atom)*
    ;

atom:   ID
    |   INT
    ;

ID  :   ('a'..'z'|'A'..'Z')+ ;

INT :   ('0'..'9')+ ;

WS  :   (' '|'\t'|'\n')+ {$channel=HIDDEN;}
    ;
