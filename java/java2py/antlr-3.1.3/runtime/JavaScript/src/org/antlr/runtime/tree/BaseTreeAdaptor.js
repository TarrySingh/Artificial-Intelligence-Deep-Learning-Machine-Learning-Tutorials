/** A TreeAdaptor that works with any Tree implementation. */
org.antlr.runtime.tree.BaseTreeAdaptor = function() {
    this.uniqueNodeID = 1;
};

org.antlr.lang.extend(org.antlr.runtime.tree.BaseTreeAdaptor,
                      org.antlr.runtime.tree.TreeAdaptor,
{
    nil: function() {
        return this.create(null);
    },

    /** create tree node that holds the start and stop tokens associated
     *  with an error.
     *
     *  If you specify your own kind of tree nodes, you will likely have to
     *  override this method. CommonTree returns Token.INVALID_TOKEN_TYPE
     *  if no token payload but you might have to set token type for diff
     *  node type.
     */
    errorNode: function(input, start, stop, e) {
        var t = new org.antlr.runtime.tree.CommonErrorNode(input, start, stop, e);
        return t;
    },

    isNil: function(tree) {
        return tree.isNil();
    },

    /** This is generic in the sense that it will work with any kind of
     *  tree (not just Tree interface).  It invokes the adaptor routines
     *  not the tree node routines to do the construction.  
     */
    dupTree: function(t, parent) {
        if (arguments.length===1) {
            parent = null;
        }
        if ( !t ) {
            return null;
        }
        var newTree = this.dupNode(t);
        // ensure new subtree root has parent/child index set
        this.setChildIndex(newTree, this.getChildIndex(t)); // same index in new tree
        this.setParent(newTree, parent);
        var n = this.getChildCount(t),
            i, child, newSubTree;
        for (i = 0; i < n; i++) {
            child = this.getChild(t, i);
            newSubTree = this.dupTree(child, t);
            this.addChild(newTree, newSubTree);
        }
        return newTree;
    },

    /** Add a child to the tree t.  If child is a flat tree (a list), make all
     *  in list children of t.  Warning: if t has no children, but child does
     *  and child isNil then you can decide it is ok to move children to t via
     *  t.children = child.children; i.e., without copying the array.  Just
     *  make sure that this is consistent with have the user will build
     *  ASTs.
     */
    addChild: function(t, child) {
        if ( t && org.antlr.lang.isValue(child) ) {
            t.addChild(child);
        }
    },

    /** If oldRoot is a nil root, just copy or move the children to newRoot.
     *  If not a nil root, make oldRoot a child of newRoot.
     *
     *    old=^(nil a b c), new=r yields ^(r a b c)
     *    old=^(a b c), new=r yields ^(r ^(a b c))
     *
     *  If newRoot is a nil-rooted single child tree, use the single
     *  child as the new root node.
     *
     *    old=^(nil a b c), new=^(nil r) yields ^(r a b c)
     *    old=^(a b c), new=^(nil r) yields ^(r ^(a b c))
     *
     *  If oldRoot was null, it's ok, just return newRoot (even if isNil).
     *
     *    old=null, new=r yields r
     *    old=null, new=^(nil r) yields ^(nil r)
     *
     *  Return newRoot.  Throw an exception if newRoot is not a
     *  simple node or nil root with a single child node--it must be a root
     *  node.  If newRoot is ^(nil x) return x as newRoot.
     *
     *  Be advised that it's ok for newRoot to point at oldRoot's
     *  children; i.e., you don't have to copy the list.  We are
     *  constructing these nodes so we should have this control for
     *  efficiency.
     */
    becomeRoot: function(newRoot, oldRoot) {
        if (newRoot instanceof org.antlr.runtime.Token || !newRoot) {
            newRoot = this.create(newRoot);
        }

        var newRootTree = newRoot,
            oldRootTree = oldRoot;
        if ( !oldRoot ) {
            return newRoot;
        }
        // handle ^(nil real-node)
        if ( newRootTree.isNil() ) {
            var nc = newRootTree.getChildCount();
            if (nc===1) {
                newRootTree = newRootTree.getChild(0);
            } else if ( nc>1 ) {
                // TODO: make tree run time exceptions hierarchy
                throw new Error("more than one node as root (TODO: make exception hierarchy)");
            }
        }
        // add oldRoot to newRoot; addChild takes care of case where oldRoot
        // is a flat list (i.e., nil-rooted tree).  All children of oldRoot
        // are added to newRoot.
        newRootTree.addChild(oldRootTree);
        return newRootTree;
    },

    /** Transform ^(nil x) to x */
    rulePostProcessing: function(root) {
        var r = root;
        if ( r && r.isNil() ) {
            if ( r.getChildCount()===0 ) {
                r = null;
            }
            else if ( r.getChildCount()===1 ) {
                r = r.getChild(0);
                // whoever invokes rule will set parent and child index
                r.setParent(null);
                r.setChildIndex(-1);
            }
        }
        return r;
    },

    create: function(tokenType, fromToken) {
        var text, t;
        if (arguments.length===2) {
            if (org.antlr.lang.isString(arguments[1])) {
                text = arguments[1];
                fromToken = this.createToken(tokenType, text);
                t = this.create(fromToken);
                return t;
            } else {
                fromToken = this.createToken(fromToken);
                fromToken.setType(tokenType);
                t = this.create(fromToken);
                return t;
            }
        } else if (arguments.length===3) {
            text = arguments[2];
            fromToken = this.createToken(fromToken);
            fromToken.setType(tokenType);
            fromToken.setText(text);
            t = this.create(fromToken);
            return t;
        }
    },

    getType: function(t) {
        t.getType();
        return 0;
    },

    setType: function(t, type) {
        throw new Error("don't know enough about Tree node");
    },

    getText: function(t) {
        return t.getText();
    },

    setText: function(t, text) {
        throw new Error("don't know enough about Tree node");
    },

    getChild: function(t, i) {
        return t.getChild(i);
    },

    setChild: function(t, i, child) {
        t.setChild(i, child);
    },

    deleteChild: function(t, i) {
        return t.deleteChild(i);
    },

    getChildCount: function(t) {
        return t.getChildCount();
    },

    getUniqueID: function(node) {
        if ( !this.treeToUniqueIDMap ) {
             this.treeToUniqueIDMap = {};
        }
        var prevID = this.treeToUniqueIDMap[node];
        if ( org.antlr.lang.isValue(prevID) ) {
            return prevID;
        }
        var ID = this.uniqueNodeID;
        this.treeToUniqueIDMap[node] = ID;
        this.uniqueNodeID++;
        return ID;
        // GC makes these nonunique:
        // return System.identityHashCode(node);
    }
});
