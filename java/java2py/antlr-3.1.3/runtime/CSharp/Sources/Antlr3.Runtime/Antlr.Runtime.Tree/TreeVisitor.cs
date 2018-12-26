
namespace Antlr.Runtime.Tree {

	/// <summary>
	/// Do a depth first walk of a tree, applying pre() and post() actions
	/// as we discover and finish nodes.
	/// </summary>
	public class TreeVisitor {
	    protected ITreeAdaptor adaptor;
	    
	    public TreeVisitor(ITreeAdaptor adaptor) {
	        this.adaptor = adaptor;
	    }
	    
	    public TreeVisitor() : this(new CommonTreeAdaptor()) { }
	    
		/// <summary>
		/// Visit every node in tree t and trigger an action for each node
	    /// before/after having visited all of its children.
	    /// Execute both actions even if t has no children.
	    /// If a child visit yields a new child, it can update its
	    /// parent's child list or just return the new child.  The
	    /// child update code works even if the child visit alters its parent
	    /// and returns the new tree.
	    /// 
	    /// Return result of applying post action to this node.
		/// </summary>
	    public object Visit(object t, ITreeVisitorAction action) {
	        bool isNil = adaptor.IsNil(t);
	        if ( action!=null && !isNil ) {
	            t = action.Pre(t); // if rewritten, walk children of new t
	        }
	        int n = adaptor.GetChildCount(t);
	        for (int i=0; i<n; i++) {
	            object child = adaptor.GetChild(t, i);
	            object visitResult = Visit(child, action);
	            object childAfterVisit = adaptor.GetChild(t, i);
	            if ( visitResult !=  childAfterVisit ) { // result & child differ?
	                adaptor.SetChild(t, i, visitResult);
	            }
	        }
	        if ( action!=null && !isNil ) t = action.Post(t);
	        return t;
	    }
	}
}
