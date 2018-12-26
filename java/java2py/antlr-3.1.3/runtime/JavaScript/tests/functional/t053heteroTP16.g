grammar t053heteroTP16;
options {
    language=JavaScript;
    output=AST;
    tokenVocab=t053heteroT16;
}
tokens { ROOT; }
@header {
function V16(ttype, x) {
    V16.superclass.constructor.call(this, new org.antlr.runtime.CommonToken(ttype));
    this.foobar = x;
};
org.antlr.lang.extend(V16, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        return t053heteroTP16Parser.tokenNames[this.getType()] + "<V>;" + this.foobar;
    }
});
}
a : ID -> ROOT<V16>[42] ID
  ;

