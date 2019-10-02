org.antlr.runtime.tree.RewriteCardinalityException = function(elementDescription) {
    this.elementDescription = elementDescription;
};

/** Base class for all exceptions thrown during AST rewrite construction.
 *  This signifies a case where the cardinality of two or more elements
 *  in a subrule are different: (ID INT)+ where |ID|!=|INT|
 */
org.antlr.lang.extend(org.antlr.runtime.tree.RewriteCardinalityException, Error, {
    getMessage: function() {
		if ( org.antlr.lang.isString(this.elementDescription) ) {
			return this.elementDescription;
		}
		return null;
	},
    name: function() {
        return "org.antlr.runtime.tree.RewriteCardinalityException";
    }
});
