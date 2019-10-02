/** Ref to ID or expr but no tokens in ID stream or subtrees in expr stream */
org.antlr.runtime.tree.RewriteEmptyStreamException = function(elementDescription) {
    var sup = org.antlr.runtime.tree.RewriteEmptyStreamException.superclass; 
    sup.constructor.call(this, elementDescription);
};

org.antlr.lang.extend(org.antlr.runtime.tree.RewriteEmptyStreamException,
                  org.antlr.runtime.tree.RewriteCardinalityException, {
    name: function() {
        return "org.antlr.runtime.tree.RewriteEmptyStreamException";
    }
});
