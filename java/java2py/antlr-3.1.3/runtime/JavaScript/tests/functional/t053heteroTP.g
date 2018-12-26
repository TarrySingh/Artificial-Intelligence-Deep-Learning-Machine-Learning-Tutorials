grammar t053heteroTP;
options {
    language=JavaScript;
    output=AST;
    tokenVocab=t053heteroT;
}
tokens { ROOT; }
@header {
function VX(ttype, tree) {
    VX.superclass.constructor.apply(this, arguments);
};
org.antlr.lang.extend(VX, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        return VX.superclass.toString.call(this) + "<V>";
    }
});
}
a : ID<V> ';'<V>;

