namespace Antlr.Runtime.Tree
{
	/// <summary>
	/// How to execute code for node t when a visitor visits node t.  Execute
	/// Pre() before visiting children and execute Post() after visiting children.
	/// </summary>
	public interface ITreeVisitorAction
	{
	    /// <summary>
	    /// Execute an action before visiting children of t.  Return t or
	    /// a rewritten t.  Children of returned value will be visited.
	    /// </summary>
	    object Pre(object t);
	
	    /// <summary>
	    /// Execute an action after visiting children of t. Return t or
	    /// a rewritten t. It is up to the visitor to decide what to do
	    /// with the return value.
	    /// </summary>
	    object Post(object t);
	}
}
