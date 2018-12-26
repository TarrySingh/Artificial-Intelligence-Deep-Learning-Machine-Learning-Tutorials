/*
 [The "BSD licence"]
 Copyright (c) 2005-2006 Terence Parr
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
	/** A generic tree implementation with no payload.  You must subclass to
	 *  actually have any user data.  ANTLR v3 uses a list of children approach
	 *  instead of the child-sibling approach in v2.  A flat tree (a list) is
	 *  an empty node whose children represent the list.  An empty, but
	 *  non-null node is called "nil".
	 */
	public class BaseTree implements Tree {
		protected var _children:Array;
	
		/** Create a new node from an existing node does nothing for BaseTree
		 *  as there are no fields other than the children list, which cannot
		 *  be copied as the children are not considered part of this node. 
		 */
		public function BaseTree(node:Tree = null) {
		}
	
		public function getChild(i:int):Tree {
			if ( _children==null || i>=_children.length ) {
				return null;
			}
			return BaseTree(_children[i]);
		}
		
		/** Get the children internal List; note that if you directly mess with
		 *  the list, do so at your own risk.
		 */
		public function get children():Array {
			return _children;
		}
		
		public function getFirstChildWithType(type:int):Tree {
			for (var i:int = 0; _children!=null && i < _children.length; i++) {
				var t:Tree = Tree(_children[i]);
				if ( t.type==type ) {
					return t;
				}
			}	
			return null;
		}
	
		public function get childCount():int {
			if ( _children==null ) {
				return 0;
			}
			return _children.length;
		}
	
		/** Add t as child of this node.
		 *
		 *  Warning: if t has no children, but child does
		 *  and child isNil then this routine moves children to t via
		 *  t.children = child.children; i.e., without copying the array.
		 */
		public function addChild(t:Tree):void {
			if ( t==null ) {
				return; // do nothing upon addChild(null)
			}
			var childTree:BaseTree = BaseTree(t);
			if ( childTree.isNil ) { // t is an empty node possibly with children
				if ( this._children!=null && this._children == childTree._children ) {
					throw new Error("attempt to add child list to itself");
				}
				// just add all of childTree's children to this
				if ( childTree._children!=null ) {
					if ( this._children!=null ) { // must copy, this has children already
						var n:int = childTree._children.length;
						for (var i:int = 0; i < n; i++) {
							var c:Tree = Tree(childTree._children[i]);
							this.children.push(c);
							// handle double-link stuff for each child of nil root
							c.parent = this;
							c.childIndex = children.length-1;
						}
					}
					else {
						// no children for this but t has children; just set pointer
						// call general freshener routine
						this._children = childTree.children;
						this.freshenParentAndChildIndexes();
					}
				}
			}
			else { // child is not nil (don't care about children)
				if ( _children==null ) {
					_children = new Array(); // create children list on demand
				}
				_children.push(t);
				childTree.parent = this;
				childTree.childIndex = children.length-1;
			}
		}
	
		/** Add all elements of kids list as children of this node */
		public function addChildren(kids:Array):void {
			for (var i:int = 0; i < kids.length; i++) {
				var t:Tree = Tree(kids[i]);
				addChild(t);
			}
		}
	
		public function setChild(i:int, t:Tree):void {
			if ( t==null ) {
				return;
			}
			if ( t.isNil ) {
				throw new Error("Can't set single child to a list");
			}
			if ( _children==null ) {
				_children = new Array();
			}
			_children[i] = t;
			t.parent = this;
			t.childIndex = i;
		}
	
		public function deleteChild(i:int):Object {
			if ( _children==null ) {
				return null;
			}
			var killed:BaseTree = BaseTree(children.remove(i));
			// walk rest and decrement their child indexes
			this.freshenParentAndChildIndexesFrom(i);
			return killed;
		}

		/** Delete children from start to stop and replace with t even if t is
		 *  a list (nil-root tree).  num of children can increase or decrease.
		 *  For huge child lists, inserting children can force walking rest of
		 *  children to set their childindex; could be slow.
		 */
		public function replaceChildren(startChildIndex:int, stopChildIndex:int, t:Object):void {
			if ( children==null ) {
				throw new Error("indexes invalid; no children in list");
			}
			var replacingHowMany:int = stopChildIndex - startChildIndex + 1;
			var replacingWithHowMany:int;
			var newTree:BaseTree = BaseTree(t);
			var newChildren:Array = null;
			// normalize to a list of children to add: newChildren
			if ( newTree.isNil ) {
				newChildren = newTree.children;
			}
			else {
				newChildren = new Array(1);
				newChildren.add(newTree);
			}
			replacingWithHowMany = newChildren.length;
			var numNewChildren:int = newChildren.length;
			var delta:int = replacingHowMany - replacingWithHowMany;
			// if same number of nodes, do direct replace
			if ( delta == 0 ) {
				var j:int = 0; // index into new children
				for (var i:int=startChildIndex; i<=stopChildIndex; i++) {
					var child:BaseTree = BaseTree(newChildren[j]);
					children[i] = child;
					child.parent = this;
					child.childIndex= i;
	                j++;
	            }
			}
			else if ( delta > 0 ) { // fewer new nodes than there were
				// set children and then delete extra
				for (j=0; j<numNewChildren; j++) {
					children[startChildIndex+j] = newChildren[j];
				}
				var indexToDelete:int = startChildIndex+numNewChildren;
				for (var c:int=indexToDelete; c<=stopChildIndex; c++) {
					// delete same index, shifting everybody down each time
					var killed:BaseTree = BaseTree(children.remove(indexToDelete));
				}
				freshenParentAndChildIndexesFrom(startChildIndex);
			}
			else { // more new nodes than were there before
				// fill in as many children as we can (replacingHowMany) w/o moving data
				for (j=0; j<replacingHowMany; j++) {
					children[startChildIndex+j] = newChildren[j];
				}
				var numToInsert:int = replacingWithHowMany-replacingHowMany;
				for (j=replacingHowMany; j<replacingWithHowMany; j++) {
					children.splice(startChildIndex+j, 0, newChildren[j]);
				}
				freshenParentAndChildIndexesFrom(startChildIndex);
			}
		}

		public function get isNil():Boolean {
			return false;
		}
	
		/** Set the parent and child index values for all child of t */
		public function freshenParentAndChildIndexes():void {
			freshenParentAndChildIndexesFrom(0);
		}
	
		public function freshenParentAndChildIndexesFrom(offset:int):void {
			var n:int = childCount;
			for (var c:int = offset; c < n; c++) {
				var child:Tree = Tree(getChild(c));
				child.childIndex = c;
				child.parent = this;
			}
		}
	
		public function sanityCheckParentAndChildIndexes():void {
			sanityCheckParentAndChildIndexesFrom(null, -1);
		}
	
		public function sanityCheckParentAndChildIndexesFrom(parent:Tree, i:int):void {
			if ( parent!=this.parent ) {
				throw new Error("parents don't match; expected "+parent+" found "+this.parent);
			}
			if ( i!=this.childIndex ) {
				throw new Error("child indexes don't match; expected "+i+" found "+this.childIndex);
			}
			var n:int = this.childCount;
			for (var c:int = 0; c < n; c++) {
				var child:CommonTree = CommonTree(this.getChild(c));
				child.sanityCheckParentAndChildIndexesFrom(this, c);
			}
		}
	
		/** BaseTree doesn't track child indexes. */
		public function get childIndex():int {
			return 0;
		}
		
		public function set childIndex(index:int):void {
		}
	
		/** BaseTree doesn't track parent pointers. */
		public function get parent():Tree {
			return null;
		}
		public function set parent(t:Tree):void {
		}

        /** Walk upwards looking for ancestor with this token type. */
        public function hasAncestor(ttype:int):Boolean { return getAncestor(ttype)!=null; }
    
        /** Walk upwards and get first ancestor with this token type. */
        public function getAncestor(ttype:int):Tree {
            var t:Tree = this;
            t = t.parent;
            while ( t!=null ) {
                if ( t.type==ttype ) return t;
                t = t.parent;
            }
            return null;
        }
    
        /** Return a list of all ancestors of this node.  The first node of
         *  list is the root and the last is the parent of this node.
         */
        public function get ancestors():Array {
            if ( parent==null ) return null;
            var ancestors:Array = new Array();
            var t:Tree = this;
            t = t.parent;
            while ( t!=null ) {
                ancestors.unshift(t); // insert at start
                t = t.parent;
            }
            return ancestors;
        }

		/** Print out a whole tree not just a node */
	    public function toStringTree():String {
			if ( _children==null || _children.length==0 ) {
				return String(this);
			}
			var buf:String = "";
			if ( !isNil ) {
				buf += "(";
				buf += String(this);
				buf += ' ';
			}
			for (var i:int = 0; _children!=null && i < _children.length; i++) {
				var t:BaseTree = BaseTree(_children[i]);
				if ( i>0 ) {
					buf += ' ';
				}
				buf += t.toStringTree();
			}
			if ( !isNil ) {
				buf += ")";
			}
			return buf;
		}
	
	    public function get line():int {
			return 0;
		}
	
		public function get charPositionInLine():int {
			return 0;
		}

		// "Abstract" functions since there are no abstract classes in actionscript
		 
		public function dupNode():Tree {
			throw new Error("Not implemented");
		}
	
		public function get type():int {
			throw new Error("Not implemented");
		}
	
		public function get text():String {
			throw new Error("Not implemented");
		}
	
		public function get tokenStartIndex():int {
			throw new Error("Not implemented");
		}
	
		public function set tokenStartIndex(index:int):void {
			throw new Error("Not implemented");
		}
	
		public function get tokenStopIndex():int {
			throw new Error("Not implemented");
		}
	
		public function set tokenStopIndex(index:int):void {
			throw new Error("Not implemented");
		}
	
	}
}