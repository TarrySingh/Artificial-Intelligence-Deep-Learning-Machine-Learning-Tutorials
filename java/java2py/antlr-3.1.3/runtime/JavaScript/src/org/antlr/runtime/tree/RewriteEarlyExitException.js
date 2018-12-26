/** No elements within a (...)+ in a rewrite rule */
org.antlr.runtime.tree.RewriteEarlyExitException = function(elementDescription) {
    var sup = org.antlr.runtime.tree.RewriteEarlyExitException.superclass;
    if (org.antlr.lang.isUndefined(elementDescription)) {
        elementDescription = null;
    }
    sup.constructor.call(this, elementDescription);
};

org.antlr.lang.extend(org.antlr.runtime.tree.RewriteEarlyExitException,
                  org.antlr.runtime.tree.RewriteCardinalityException, {
    name: function() {
        return "org.antlr.runtime.tree.RewriteEarlyExitException";
    }    
});
