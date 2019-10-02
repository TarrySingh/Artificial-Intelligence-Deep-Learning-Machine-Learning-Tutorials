grammar t053heteroT9;
options {
    language=JavaScript;
    output=AST;
}
@header {
function V2() {
    var x, y, z, token, ttype;
    if (arguments.length===4) {
        ttype = arguments[0];
        x = arguments[1];
        y = arguments[2];
        z = arguments[3];
        token = new org.antlr.runtime.CommonToken(ttype, "");
    } else if (arguments.length===3) {
        ttype = arguments[0];
        token = arguments[1];
        x = arguments[2];
        y = 0;
        z = 0;
    } else {
        throw new Error("Invalid args");
    }

    V2.superclass.constructor.call(this, token);
    this.x = x;
    this.y = y;
    this.z = z;
};

org.antlr.lang.extend(V2, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        var txt = "";
        if (this.token) {
            txt += this.getText();
        }
        txt += "<V>;"+this.x.toString()+this.y.toString()+this.z.toString();
        return txt;
    }
});
}
a : ID -> ID<V2>[42,19,30] ID<V2>[$ID,99];
ID : 'a'..'z'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;

