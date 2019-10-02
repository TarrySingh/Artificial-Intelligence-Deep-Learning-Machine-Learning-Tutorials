org.antlr.runtime.tree.RewriteRuleSubtreeStream = function() {
    var sup = org.antlr.runtime.tree.RewriteRuleSubtreeStream.superclass;
    sup.constructor.apply(this, arguments);
};

org.antlr.lang.extend(org.antlr.runtime.tree.RewriteRuleSubtreeStream,
                  org.antlr.runtime.tree.RewriteRuleElementStream, {
	/** Treat next element as a single node even if it's a subtree.
	 *  This is used instead of next() when the result has to be a
	 *  tree root node.  Also prevents us from duplicating recently-added
	 *  children; e.g., ^(type ID)+ adds ID to type and then 2nd iteration
	 *  must dup the type node, but ID has been added.
	 *
	 *  Referencing a rule result twice is ok; dup entire tree as
	 *  we can't be adding trees as root; e.g., expr expr.
	 *
	 *  Hideous code duplication here with super.next().  Can't think of
	 *  a proper way to refactor.  This needs to always call dup node
	 *  and super.next() doesn't know which to call: dup node or dup tree.
	 */
    nextNode: function() {
		var n = this.size(),
            el;
		if ( this.dirty || (this.cursor>=n && n===1) ) {
			// if out of elements and size is 1, dup (at most a single node
			// since this is for making root nodes).
			el = this._next();
			return this.adaptor.dupNode(el);
		}
		// test size above then fetch
		el = this._next();
		return el;
	},

    dup: function(el) {
		return this.adaptor.dupTree(el);
	}
});