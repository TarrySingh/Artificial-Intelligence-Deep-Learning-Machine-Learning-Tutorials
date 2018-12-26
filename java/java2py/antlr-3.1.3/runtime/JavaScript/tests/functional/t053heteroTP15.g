grammar t053heteroTP15;
options {
    language=JavaScript;
    output=AST;
    tokenVocab=t053heteroT15;
    ASTLabelType=CommonTree;
}
tokens { ROOT; }
@header {
function V15(ttype) {
    V15.superclass.constructor.call(this, new org.antlr.runtime.CommonToken(ttype));
};
org.antlr.lang.extend(V15, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        return t053heteroTP15Parser.tokenNames[this.getType()] + "<V>";
    }
});
}
a : ID -> ROOT<V15> ID
  ;

