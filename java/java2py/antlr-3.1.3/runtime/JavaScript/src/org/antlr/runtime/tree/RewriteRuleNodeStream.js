/** Queues up nodes matched on left side of -> in a tree parser. This is
 *  the analog of RewriteRuleTokenStream for normal parsers. 
 */
org.antlr.runtime.tree.RewriteRuleNodeStream = function(adaptor, elementDescription, el) {
    org.antlr.runtime.tree.RewriteRuleNodeStream.superclass.constructor.apply(this, arguments);
};

org.antlr.lang.extend(org.antlr.runtime.tree.RewriteRuleNodeStream,
                  org.antlr.runtime.tree.RewriteRuleElementStream,
{
    nextNode: function() {
        return this._next();
    },

    toTree: function(el) {
        return this.adaptor.dupNode(el);
    },

    dup: function() {
        // we dup every node, so don't have to worry about calling dup; short-
        // circuited next() so it doesn't call.
        throw new Error("dup can't be called for a node stream.");
    }
});
