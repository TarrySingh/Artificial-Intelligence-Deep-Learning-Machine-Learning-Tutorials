grammar t053heteroT10;
options {
    language=JavaScript;
    output=AST;
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
}
a : ID INT -> ^(ID<V> INT) ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+ ;
WS : (' '|'\n') {$channel=HIDDEN;} ;


