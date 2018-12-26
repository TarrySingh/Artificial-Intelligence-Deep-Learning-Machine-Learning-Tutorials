package org.antlr.runtime.tree;

/** Do a depth first walk of a tree, applying pre() and post() actions
 *  as we discover and finish nodes.
 */
public class TreeVisitor {
    protected TreeAdaptor adaptor;
    
    public TreeVisitor(TreeAdaptor adaptor) {
        this.adaptor = adaptor;
    }
    public TreeVisitor() { this(new CommonTreeAdaptor()); }
    
    /** Visit every node in tree t and trigger an action for each node
     *  before/after having visited all of its children.
     *  Execute both actions even if t has no children.
     *  If a child visit yields a new child, it can update its
     *  parent's child list or just return the new child.  The
     *  child update code works even if the child visit alters its parent
     *  and returns the new tree.
     *
     *  Return result of applying post action to this node.
     */
    public Object visit(Object t, TreeVisitorAction action) {
        // System.out.println("visit "+((Tree)t).toStringTree());
        boolean isNil = adaptor.isNil(t);
        if ( action!=null && !isNil ) {
            t = action.pre(t); // if rewritten, walk children of new t
        }
        int n = adaptor.getChildCount(t);
        for (int i=0; i<n; i++) {
            Object child = adaptor.getChild(t, i);
            Object visitResult = visit(child, action);
            Object childAfterVisit = adaptor.getChild(t, i);
            if ( visitResult !=  childAfterVisit ) { // result & child differ?
                adaptor.setChild(t, i, visitResult);
            }
        }
        if ( action!=null && !isNil ) t = action.post(t);
        return t;
    }
}
