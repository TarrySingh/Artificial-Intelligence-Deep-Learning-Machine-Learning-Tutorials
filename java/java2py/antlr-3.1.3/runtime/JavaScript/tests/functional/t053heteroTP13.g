grammar t053heteroTP13;
options {
    language=JavaScript;
    output=AST;
    tokenVocab=t053heteroT13;
}
@header {
function V() {
    V.superclass.constructor.apply(this, arguments);
};
org.antlr.lang.extend(V, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        return this.getText() + "<V>";
    }
});

function W() {
    W.superclass.constructor.apply(this, arguments);
};
org.antlr.lang.extend(W, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        return this.getText() + "<W>";
    }
});
}
a : ID INT -> INT<V> ID<W>
  ;

