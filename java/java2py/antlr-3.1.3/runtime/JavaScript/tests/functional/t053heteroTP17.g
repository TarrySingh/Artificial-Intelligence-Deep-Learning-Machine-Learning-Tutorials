grammar t053heteroTP17;
options {
    language=JavaScript;
    output=AST;
    tokenVocab=t053heteroT17;
}
tokens { ROOT; }
@header {
function V17(ttype) {
    V17.superclass.constructor.call(this, new org.antlr.runtime.CommonToken(ttype));
};
org.antlr.lang.extend(V17, org.antlr.runtime.tree.CommonTree, {
    toString: function() {
        return t053heteroTP17Parser.tokenNames[this.getType()] + "<V>";
    }
});
}
a : ID -> ^(ROOT<V17> ID)
  ;

