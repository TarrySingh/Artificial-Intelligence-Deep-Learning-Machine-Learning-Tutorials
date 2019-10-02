org.antlr.runtime.tree.RewriteRuleTokenStream = function(adaptor, elementDescription, el) {
    var sup = org.antlr.runtime.tree.RewriteRuleTokenStream.superclass;
    sup.constructor.apply(this, arguments);
};

org.antlr.lang.extend(org.antlr.runtime.tree.RewriteRuleTokenStream,
                  org.antlr.runtime.tree.RewriteRuleElementStream, {
    /** Get next token from stream and make a node for it */
    nextNode: function() {
        var t = this._next();
        return this.adaptor.create(t);
    },

    nextToken: function() {
        return this._next();
    },

    /** Don't convert to a tree unless they explicitly call nextTree.
     *  This way we can do hetero tree nodes in rewrite.
     */
    toTree: function(el) {
        return el;
    },

    dup: function(el) {
        throw new Error("dup can't be called for a token stream.");
    }
});
