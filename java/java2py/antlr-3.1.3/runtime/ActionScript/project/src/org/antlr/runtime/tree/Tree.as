package org.antlr.runtime.tree {
	/** What does a tree look like?  ANTLR has a number of support classes
	 *  such as CommonTreeNodeStream that work on these kinds of trees.  You
	 *  don't have to make your trees implement this interface, but if you do,
	 *  you'll be able to use more support code.
	 *
	 *  NOTE: When constructing trees, ANTLR can build any kind of tree; it can
	 *  even use Token objects as trees if you add a child list to your tokens.
	 *
	 *  This is a tree node without any payload; just navigation and factory stuff.
	 */
	public interface Tree {
	
		function getChild(i:int):Tree;
	
		function get childCount():int;
	
		// Tree tracks parent and child index now > 3.0
	
		function get parent():Tree;
	
		function set parent(t:Tree):void;
		
		/** Is there is a node above with token type ttype? */
        function hasAncestor(ttype:int):Boolean;
    
        /** Walk upwards and get first ancestor with this token type. */
        function getAncestor(ttype:int):Tree;
    
        /** Return a list of all ancestors of this node.  The first node of
         *  list is the root and the last is the parent of this node.
         */
        function get ancestors():Array;

	
		/** This node is what child index? 0..n-1 */
		function get childIndex():int;
	
		function set childIndex(index:int):void;
	
		/** Set the parent and child index values for all children */
		function freshenParentAndChildIndexes():void;
	
		/** Add t as a child to this node.  If t is null, do nothing.  If t
		 *  is nil, add all children of t to this' children.
		 */
		function addChild(t:Tree):void;
	
		/** Set ith child (0..n-1) to t; t must be non-null and non-nil node */
		function setChild(i:int, t:Tree):void;
	
		function deleteChild(i:int):Object;
	
		/** Delete children from start to stop and replace with t even if t is
		 *  a list (nil-root tree).  num of children can increase or decrease.
		 *  For huge child lists, inserting children can force walking rest of
		 *  children to set their childindex; could be slow.
		 */
		function replaceChildren(startChildIndex:int, stopChildIndex:int, t:Object):void;	

		/** Indicates the node is a nil node but may still have children, meaning
		 *  the tree is a flat list.
		 */
		function get isNil():Boolean;
	
		/**  What is the smallest token index (indexing from 0) for this node
		 *   and its children?
		 */
		function get tokenStartIndex():int;
	
		function set tokenStartIndex(index:int):void;
	
		/**  What is the largest token index (indexing from 0) for this node
		 *   and its children?
		 */
		function get tokenStopIndex():int;
	
		function set tokenStopIndex(index:int):void;
	
		function dupNode():Tree;
	
		/** Return a token type; needed for tree parsing */
		function get type():int;
	
		function get text():String;
	
		/** In case we don't have a token payload, what is the line for errors? */
		function get line():int;
	
		function get charPositionInLine():int;
	
		function toStringTree():String;

	}

}