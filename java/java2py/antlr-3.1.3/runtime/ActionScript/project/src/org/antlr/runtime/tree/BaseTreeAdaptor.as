package org.antlr.runtime.tree {
    
	import flash.utils.Dictionary;
	
	import mx.utils.ObjectUtil;
	
	import org.antlr.runtime.*;

	/** A TreeAdaptor that works with any Tree implementation. */	
	public class BaseTreeAdaptor implements TreeAdaptor {
		/** System.identityHashCode() is not always unique; we have to
		 *  track ourselves.  That's ok, it's only for debugging, though it's
		 *  expensive: we have to create a hashtable with all tree nodes in it.
		 */
		protected var treeToUniqueIDMap:Dictionary;
		protected var uniqueNodeID:int = 1;
	
		public function nil():Object {
			return createWithPayload(null);
		}
	
		/** create tree node that holds the start and stop tokens associated
    	 *  with an error.
    	 *
    	 *  If you specify your own kind of tree nodes, you will likely have to
    	 *  override this method. CommonTree returns Token.INVALID_TOKEN_TYPE
    	 *  if no token payload but you might have to set token type for diff
    	 *  node type.
    	 * 
         *  You don't have to subclass CommonErrorNode; you will likely need to
         *  subclass your own tree node class to avoid class cast exception.
    	 */
    	public function errorNode(input:TokenStream, start:Token, stop:Token,
    							e:RecognitionException):Object {
    		var t:CommonErrorNode = new CommonErrorNode(input, start, stop, e);
    		//System.out.println("returning error node '"+t+"' @index="+input.index());
    		return t;
    	}

		public function isNil(tree:Object):Boolean {
			return Tree(tree).isNil;
		}
	
		public function dupTree(tree:Object):Object {
			return dupTreeWithParent(tree, null);
		}
	
		/** This is generic in the sense that it will work with any kind of
		 *  tree (not just Tree interface).  It invokes the adaptor routines
		 *  not the tree node routines to do the construction.  
		 */
		public function dupTreeWithParent(t:Object, parent:Object):Object {
			if ( t==null ) {
				return null;
			}
			var newTree:Object = dupNode(t);
			// ensure new subtree root has parent/child index set
			setChildIndex(newTree, getChildIndex(t)); // same index in new tree
			setParent(newTree, parent);
			var n:int = getChildCount(t);
			for (var i:int = 0; i < n; i++) {
				var child:Object = getChild(t, i);
				var newSubTree:Object = dupTreeWithParent(child, t);
				addChild(newTree, newSubTree);
			}
			return newTree;
		}
		
		/** Add a child to the tree t.  If child is a flat tree (a list), make all
		 *  in list children of t.  Warning: if t has no children, but child does
		 *  and child isNil then you can decide it is ok to move children to t via
		 *  t.children = child.children; i.e., without copying the array.  Just
		 *  make sure that this is consistent with have the user will build
		 *  ASTs.
		 */
		public function addChild(t:Object, child:Object):void {
			if ( t!=null && child!=null ) {
				Tree(t).addChild(Tree(child));
			}
		}
	
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
		public function becomeRoot(newRoot:Object, oldRoot:Object):Object {			
			// If new Root is token, then create a Tree.
			if (newRoot is Token) {
				newRoot = createWithPayload(Token(newRoot));
			}

			var newRootTree:Tree = Tree(newRoot);
			var oldRootTree:Tree = Tree(oldRoot);
			if ( oldRoot==null ) {
				return newRoot;
			}
			// handle ^(nil real-node)
			if ( newRootTree.isNil ) {
				var nc:int = newRootTree.childCount;
	            if ( nc==1 ) newRootTree = Tree(newRootTree.getChild(0));
	            else if ( nc >1 ) {
					// TODO: make tree run time exceptions hierarchy
					throw new Error("more than one node as root (TODO: make exception hierarchy)");
				}
			}
			// add oldRoot to newRoot; addChild takes care of case where oldRoot
			// is a flat list (i.e., nil-rooted tree).  All children of oldRoot
			// are added to newRoot.
			newRootTree.addChild(oldRootTree);
			return newRootTree;
		}
	
		/** Transform ^(nil x) to x and nil to null */
		public function rulePostProcessing(root:Object):Object {
			var r:Tree = Tree(root);
			if ( r!=null && r.isNil ) {
				if ( r.childCount==0 ) {
					r = null;
				}
				else if ( r.childCount==1 ) {
					r = Tree(r.getChild(0));
					// whoever invokes rule will set parent and child index
					r.parent = null;
					r.childIndex = -1;
				}
			}
			return r;
		}
	
		public function createFromToken(tokenType:int, fromToken:Token, text:String = null):Object {
			fromToken = createToken(fromToken);
			fromToken.type = tokenType;
			if (text != null) {
				fromToken.text = text;
			}
			return createWithPayload(fromToken);
		}
	
		public function createFromType(tokenType:int, text:String):Object {
			var fromToken:Token = createTokenFromType(tokenType, text);
			return createWithPayload(fromToken);
		}
	
		public function getType(t:Object):int {
			return Tree(t).type;
		}
	
		public function setType(t:Object, type:int):void {
			throw new Error("don't know enough about Tree node");
		}
	
		public function getText(t:Object):String {
			return Tree(t).text;
		}
	
		public function setText(t:Object, text:String):void {
			throw new Error("don't know enough about Tree node");
		}
	
		public function getChild(t:Object, i:int):Object {
			return Tree(t).getChild(i);
		}

		public function setChild(t:Object, i:int, child:Object):void {
			Tree(t).setChild(i, Tree(child));
		}
	
		public function deleteChild(t:Object, i:int):Object {
			return Tree(t).deleteChild(i);
		}
		
		public function getChildCount(t:Object):int {
			return Tree(t).childCount;
		}
	
		public function getUniqueID(node:Object):int {
			if ( treeToUniqueIDMap==null ) {
				 treeToUniqueIDMap = new Dictionary();
			}
			if (treeToUniqueIDMap.hasOwnProperty(node)) {
				return treeToUniqueIDMap[node];
			}

			var ID:int = uniqueNodeID;
			treeToUniqueIDMap[node] = ID;
			uniqueNodeID++;
			return ID;

		}
	
		/** Tell me how to create a token for use with imaginary token nodes.
		 *  For example, there is probably no input symbol associated with imaginary
		 *  token DECL, but you need to create it as a payload or whatever for
		 *  the DECL node as in ^(DECL type ID).
		 *
		 *  If you care what the token payload objects' type is, you should
		 *  override this method and any other createToken variant.
		 */
		public function createTokenFromType(tokenType:int, text:String):Token {
			throw new Error("Not implemented - abstract function");
		}
	
		/** Tell me how to create a token for use with imaginary token nodes.
		 *  For example, there is probably no input symbol associated with imaginary
		 *  token DECL, but you need to create it as a payload or whatever for
		 *  the DECL node as in ^(DECL type ID).
		 *
		 *  This is a variant of createToken where the new token is derived from
		 *  an actual real input token.  Typically this is for converting '{'
		 *  tokens to BLOCK etc...  You'll see
		 *
		 *    r : lc='{' ID+ '}' -> ^(BLOCK[$lc] ID+) ;
		 *
		 *  If you care what the token payload objects' type is, you should
		 *  override this method and any other createToken variant.
		 * 
		 */
		public function createToken(fromToken:Token):Token {
			throw new Error("Not implemented - abstract function");
		}
		
		public function createWithPayload(payload:Token):Object {
			throw new Error("Not implemented - abstract function");
		}
		
		public function dupNode(t:Object):Object {
			throw new Error("Not implemented - abstract function");
		}
	
		public function getToken(t:Object):Token {
			throw new Error("Not implemented - abstract function");
		}
	
		public function setTokenBoundaries(t:Object, startToken:Token, stopToken:Token):void {
			throw new Error("Not implemented - abstract function");
		}
	
		public function getTokenStartIndex(t:Object):int {
			throw new Error("Not implemented - abstract function");
		}
	
		public function getTokenStopIndex(t:Object):int {
			throw new Error("Not implemented - abstract function");
		}

		public function getParent(t:Object):Object {
			throw new Error("Not implemented - abstract function");
		}
		
		public function setParent(t:Object, parent:Object):void {
			throw new Error("Not implemented - abstract function");
		}

		public function getChildIndex(t:Object):int {
			throw new Error("Not implemented - abstract function");
		}
		
		public function setChildIndex(t:Object, index:int):void {
			throw new Error("Not implemented - abstract function");
		}
	
		public function replaceChildren(parent:Object, startChildIndex:int, stopChildIndex:int, t:Object):void {
			throw new Error("Not implemented - abstract function");
		}	
		
		public function create(... args):Object {
			if (args.length == 1 && args[0] is Token) {
				return createWithPayload(args[0]);
			}
			else if (args.length == 2 &&
					 args[0] is int &&
					 args[1] is Token) {
			 	return createFromToken(args[0], args[1]);
		 	}
			else if (args.length == 3 &&
					 args[0] is int &&
					 args[1] is Token &&
					 args[2] is String) {
			 	return createFromToken(args[0], args[1], args[2]);
			}
			else if (args.length == 2 &&
					 args[0] is int &&
					 args[1] is String) {
			 	return createFromType(args[0], args[1]);
			} 					 
			throw new Error("No methods signature for arguments : " + ObjectUtil.toString(args));
		}
	}


}