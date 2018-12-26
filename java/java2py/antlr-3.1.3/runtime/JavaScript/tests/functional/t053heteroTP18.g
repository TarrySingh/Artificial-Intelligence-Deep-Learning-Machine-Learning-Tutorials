grammar t053heteroTP18;
options {
    language=JavaScript;
    output=AST;
    tokenVocab=t053heteroT18;
}
tokens { ROOT; }
@header {
function V18(ttype, tree) {
    if (!tree) {
        V18.superclass.constructor.call(this, new org.antlr.runtime.CommonToken(ttype));
    } else {
        V18.superclass.constructor.call(this, tree);
        this.token.type = ttype;
    }
};
org.antlr.lang.extend(V18, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        return t053heteroTP18Parser.tokenNames[this.getType()] + "<V>@" +
            this.token.getLine();
    }
});
}
a : ID -> ROOT<V18>[$ID]
  ;

