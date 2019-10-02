/*
 [The "BSD licence"]
 Copyright (c) 2005-2007 Terence Parr
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
package org.antlr.runtime.tree {
	
	import org.antlr.runtime.*;
	
	/** How to create and navigate trees.  Rather than have a separate factory
	 *  and adaptor, I've merged them.  Makes sense to encapsulate.
	 *
	 *  This takes the place of the tree construction code generated in the
	 *  generated code in 2.x and the ASTFactory.
	 *
	 *  I do not need to know the type of a tree at all so they are all
	 *  generic Objects.  This may increase the amount of typecasting needed. :(
	 */
	public interface TreeAdaptor {
		// C o n s t r u c t i o n
	
		/** Create a tree node from Token object; for CommonTree type trees,
		 *  then the token just becomes the payload.  This is the most
		 *  common create call.
		 * 
		 * 	Override if you want another kind of node to be built.
	     */
		function createWithPayload(payload:Token):Object;
	
		/** Duplicate a single tree node.
	 	 *  Override if you want another kind of node to be built.
	 	 */
	 	function dupNode(treeNode:Object):Object;
	 	
		/** Duplicate tree recursively, using dupNode() for each node */
		function dupTree(tree:Object):Object;
	
		/** Return a nil node (an empty but non-null node) that can hold
		 *  a list of element as the children.  If you want a flat tree (a list)
		 *  use "t=adaptor.nil(); t.addChild(x); t.addChild(y);"
		 */
		function nil():Object;
	
    	/** Return a tree node representing an error.  This node records the
    	 *  tokens consumed during error recovery.  The start token indicates the
    	 *  input symbol at which the error was detected.  The stop token indicates
    	 *  the last symbol consumed during recovery.
    	 *
    	 *  You must specify the input stream so that the erroneous text can
    	 *  be packaged up in the error node.  The exception could be useful
    	 *  to some applications; default implementation stores ptr to it in
    	 *  the CommonErrorNode.
    	 *
    	 *  This only makes sense during token parsing, not tree parsing.
    	 *  Tree parsing should happen only when parsing and tree construction
    	 *  succeed.
    	 */
    	function errorNode(input:TokenStream, start:Token, stop:Token, e:RecognitionException):Object;

		/** Is tree considered a nil node used to make lists of child nodes? */ 
		function isNil(tree:Object):Boolean;
	
		/** Add a child to the tree t.  If child is a flat tree (a list), make all
		 *  in list children of t.  Warning: if t has no children, but child does
		 *  and child isNil then you can decide it is ok to move children to t via
		 *  t.children = child.children; i.e., without copying the array.  Just
		 *  make sure that this is consistent with have the user will build
		 *  ASTs.  Do nothing if t or child is null.
		 */
		function addChild(t:Object, child:Object):void;
	
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
		function becomeRoot(newRoot:Object, oldRoot:Object):Object;
	
		/** Given the root of the subtree created for this rule, post process
		 *  it to do any simplifications or whatever you want.  A required
		 *  behavior is to convert ^(nil singleSubtree) to singleSubtree
		 *  as the setting of start/stop indexes relies on a single non-nil root
		 *  for non-flat trees.
		 *
		 *  Flat trees such as for lists like "idlist : ID+ ;" are left alone
		 *  unless there is only one ID.  For a list, the start/stop indexes
		 *  are set in the nil node.
		 *
		 *  This method is executed after all rule tree construction and right
		 *  before setTokenBoundaries().
		 */
		function rulePostProcessing(root:Object):Object;
	
		/** For identifying trees.
		 *
		 *  How to identify nodes so we can say "add node to a prior node"?
		 *  Even becomeRoot is an issue.  Use System.identityHashCode(node)
		 *  usually.
		 */
		function getUniqueID(node:Object):int;
	
	
		// R e w r i t e  R u l e s
	
		/** Create a new node derived from a token, with a new token type.
		 *  This is invoked from an imaginary node ref on right side of a
		 *  rewrite rule as IMAG[$tokenLabel] or IMAG[$tokenLabel, "IMAG"].
		 *
		 *  This should invoke createToken(Token).
		 */
		function createFromToken(tokenType:int, fromToken:Token, text:String = null):Object;
	
		/** Create a new node derived from a token, with a new token type.
		 *  This is invoked from an imaginary node ref on right side of a
		 *  rewrite rule as IMAG["IMAG"].
		 *
		 *  This should invoke createToken(int,String).
		 */
		function createFromType(tokenType:int, text:String):Object;
	
	
		// C o n t e n t
	
		/** For tree parsing, I need to know the token type of a node */
		function getType(t:Object):int;
	
		/** Node constructors can set the type of a node */
		function setType(t:Object, type:int):void;
	
		function getText(t:Object):String;
	
		/** Node constructors can set the text of a node */
		function setText(t:Object, text:String):void;
	
		/** Return the token object from which this node was created.
		 *  Currently used only for printing an error message.
		 *  The error display routine in BaseRecognizer needs to
		 *  display where the input the error occurred. If your
		 *  tree of limitation does not store information that can
		 *  lead you to the token, you can create a token filled with
		 *  the appropriate information and pass that back.  See
		 *  BaseRecognizer.getErrorMessage().
		 */
		function getToken(t:Object):Token;
	
		/** Where are the bounds in the input token stream for this node and
		 *  all children?  Each rule that creates AST nodes will call this
		 *  method right before returning.  Flat trees (i.e., lists) will
		 *  still usually have a nil root node just to hold the children list.
		 *  That node would contain the start/stop indexes then.
		 */
		function setTokenBoundaries(t:Object, startToken:Token, stopToken:Token):void;
	
		/** Get the token start index for this subtree; return -1 if no such index */
		function getTokenStartIndex(t:Object):int;
	
		/** Get the token stop index for this subtree; return -1 if no such index */
		function getTokenStopIndex(t:Object):int;
	
	
		// N a v i g a t i o n  /  T r e e  P a r s i n g
	
		/** Get a child 0..n-1 node */
		function getChild(t:Object, i:int):Object;
		
		/** Set ith child (0..n-1) to t; t must be non-null and non-nil node */
		function setChild(t:Object, i:int, child:Object):void;
	
		/** Remove ith child and shift children down from right. */
		function deleteChild(t:Object, i:int):Object;
	
		/** How many children?  If 0, then this is a leaf node */
		function getChildCount(t:Object):int;
		
		/** Who is the parent node of this node; if null, implies node is root.
		 *  If your node type doesn't handle this, it's ok but the tree rewrites
		 *  in tree parsers need this functionality.
		 */
		function getParent(t:Object):Object;
		function setParent(t:Object, parent:Object):void;
	
		/** What index is this node in the child list? Range: 0..n-1
		 *  If your node type doesn't handle this, it's ok but the tree rewrites
		 *  in tree parsers need this functionality.
		 */
		function getChildIndex(t:Object):int;
		function setChildIndex(t:Object, index:int):void;
	
		/** Replace from start to stop child index of parent with t, which might
		 *  be a list.  Number of children may be different
		 *  after this call.
		 *
		 *  If parent is null, don't do anything; must be at root of overall tree.
		 *  Can't replace whatever points to the parent externally.  Do nothing.
		 */
		function replaceChildren(parent:Object, startChildIndex:int, stopChildIndex:int, t:Object):void;	
		
		
		// Code - generator support - TODO place in separate namespace
		
		/**
		 * Private method used by generated code.  Based on type and number of arguments will call one of:
		 * 
		 * 	* createWithPayload
		 *  * createFromToken
		 *  * createFromType 
		 */
		function create(... args):Object;
	}
}