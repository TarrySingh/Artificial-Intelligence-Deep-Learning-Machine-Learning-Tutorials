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

	import org.antlr.runtime.TokenConstants;
	import org.antlr.runtime.TokenStream;
	

	/** A buffered stream of tree nodes.  Nodes can be from a tree of ANY kind.
	 *
	 *  This node stream sucks all nodes out of the tree specified in
	 *  the constructor during construction and makes pointers into
	 *  the tree using an array of Object pointers. The stream necessarily
	 *  includes pointers to DOWN and UP and EOF nodes.
	 *
	 *  This stream knows how to mark/release for backtracking.
	 *
	 *  This stream is most suitable for tree interpreters that need to
	 *  jump around a lot or for tree parsers requiring speed (at cost of memory).
	 *  There is some duplicated functionality here with UnBufferedTreeNodeStream
	 *  but just in bookkeeping, not tree walking etc...
	 *
	 *  @see UnBufferedTreeNodeStream
	 */
	public class CommonTreeNodeStream implements TreeNodeStream {
		public static const DEFAULT_INITIAL_BUFFER_SIZE:int = 100;
		public static const INITIAL_CALL_STACK_SIZE:int = 10;

		// all these navigation nodes are shared and hence they
		// cannot contain any line/column info
	
		protected var down:Object;
		protected var up:Object;
		protected var eof:Object;
	
		/** The complete mapping from stream index to tree node.
		 *  This buffer includes pointers to DOWN, UP, and EOF nodes.
		 *  It is built upon ctor invocation.  The elements are type
		 *  Object as we don't what the trees look like.
		 *
		 *  Load upon first need of the buffer so we can set token types
		 *  of interest for reverseIndexing.  Slows us down a wee bit to
		 *  do all of the if p==-1 testing everywhere though.
		 */
		protected var nodes:Array;
	
		/** Pull nodes from which tree? */
		protected var root:Object;
	
		/** IF this tree (root) was created from a token stream, track it. */
		protected var tokens:TokenStream;
	
		/** What tree adaptor was used to build these trees */
		internal var adaptor:TreeAdaptor;
	
		/** Reuse same DOWN, UP navigation nodes unless this is true */
		protected var uniqueNavigationNodes:Boolean = false;
	
		/** The index into the nodes list of the current node (next node
		 *  to consume).  If -1, nodes array not filled yet.
		 */
		protected var p:int = -1;
	
		/** Track the last mark() call result value for use in rewind(). */
		protected var lastMarker:int;
	
		/** Stack of indexes used for push/pop calls */
		protected var calls:Array;
	
		public function CommonTreeNodeStream(tree:Object, adaptor:TreeAdaptor = null, initialBufferSize:int = DEFAULT_INITIAL_BUFFER_SIZE) {
		    if (tree == null) {
		        // return uninitalized for static resuse function
		        return;
		    }
			this.root = tree;
			this.adaptor = adaptor == null ? new CommonTreeAdaptor() : adaptor;
			
			nodes = new Array();
			down = this.adaptor.createFromType(TokenConstants.DOWN, "DOWN");
			up = this.adaptor.createFromType(TokenConstants.UP, "UP");
			eof = this.adaptor.createFromType(TokenConstants.EOF, "EOF");
		}
	
        /** Reuse an existing node stream's buffer of nodes.  Do not point at a
         *  node stream that can change.  Must have static node list.  start/stop
         *  are indexes into the parent.nodes stream.  We avoid making a new
         *  list of nodes like this.
         */
        public static function reuse(parent:CommonTreeNodeStream, start:int, stop:int):CommonTreeNodeStream {
            var stream:CommonTreeNodeStream = new CommonTreeNodeStream(null);
            stream.root = parent.root;
            stream.adaptor = parent.adaptor;
            stream.nodes = parent.nodes.slice(start, stop);
            stream.down = parent.down;
            stream.up = parent.up;
            stream.eof = parent.eof;
            return stream;    
        }
        
		/** Walk tree with depth-first-search and fill nodes buffer.
		 *  Don't do DOWN, UP nodes if its a list (t is isNil).
		 */
		protected function fillBuffer():void {
			fillBufferTo(root);
			//System.out.println("revIndex="+tokenTypeToStreamIndexesMap);
			p = 0; // buffer of nodes intialized now
		}
	
		public function fillBufferTo(t:Object):void {
			var nil:Boolean = adaptor.isNil(t);
			if ( !nil ) {
				nodes.push(t); // add this node
			}
			// add DOWN node if t has children
			var n:int = adaptor.getChildCount(t);
			if ( !nil && n>0 ) {
				addNavigationNode(TokenConstants.DOWN);
			}
			// and now add all its children
			for (var c:int=0; c<n; c++) {
				var child:Object = adaptor.getChild(t,c);
				fillBufferTo(child);
			}
			// add UP node if t has children
			if ( !nil && n>0 ) {
				addNavigationNode(TokenConstants.UP);
			}
		}
	
		/** What is the stream index for node? 0..n-1
		 *  Return -1 if node not found.
		 */
		protected function getNodeIndex(node:Object):int {
			if ( p==-1 ) {
				fillBuffer();
			}
			for (var i:int = 0; i < nodes.length; i++) {
				var t:Object = nodes[i];
				if ( t===node ) {
					return i;
				}
			}
			return -1;
		}
			
		/** As we flatten the tree, we use UP, DOWN nodes to represent
		 *  the tree structure.  When debugging we need unique nodes
		 *  so instantiate new ones when uniqueNavigationNodes is true.
		 */
		protected function addNavigationNode(ttype:int):void {
			var navNode:Object = null;
			if ( ttype==TokenConstants.DOWN ) {
				if ( hasUniqueNavigationNodes) {
					navNode = adaptor.createFromType(TokenConstants.DOWN, "DOWN");
				}
				else {
					navNode = down;
				}
			}
			else {
				if ( hasUniqueNavigationNodes ) {
					navNode = adaptor.createFromType(TokenConstants.UP, "UP");
				}
				else {
					navNode = up;
				}
			}
			nodes.push(navNode);
		}
	
		public function getNode(i:int):Object {
			if ( p==-1 ) {
				fillBuffer();
			}
			return nodes[i];
		}
	
		public function LT(k:int):Object {
			if ( p==-1 ) {
				fillBuffer();
			}
			if ( k==0 ) {
				return null;
			}
			if ( k<0 ) {
				return LB(-k);
			}
			//System.out.print("LT(p="+p+","+k+")=");
			if ( (p+k-1) >= nodes.length ) {
				return eof;
			}
			return nodes[p+k-1];
		}
		
		public function get currentSymbol():Object { return LT(1); }
		
		/** Look backwards k nodes */
		protected function LB(k:int):Object {
			if ( k==0 ) {
				return null;
			}
			if ( (p-k)<0 ) {
				return null;
			}
			return nodes[p-k];
		}
	
		public function get treeSource():Object {
			return root;
		}
		
		public function get sourceName():String {
			return tokenStream.sourceName;
		}
	
		public function get tokenStream():TokenStream {
			return tokens;
		}
	
		public function set tokenStream(tokens:TokenStream):void {
			this.tokens = tokens;
		}
	
		public function get treeAdaptor():TreeAdaptor {
			return adaptor;
		}
	
		public function set treeAdaptor(adaptor:TreeAdaptor):void {
			this.adaptor = adaptor;
		}
		
		public function get hasUniqueNavigationNodes():Boolean {
			return uniqueNavigationNodes;
		}
	
		public function set hasUniqueNavigationNodes(uniqueNavigationNodes:Boolean):void {
			this.uniqueNavigationNodes = uniqueNavigationNodes;
		}
	
		public function consume():void {
			if ( p==-1 ) {
				fillBuffer();
			}
			p++;
		}
	
		public function LA(i:int):int {
			return adaptor.getType(LT(i));
		}
	
		public function mark():int {
			if ( p==-1 ) {
				fillBuffer();
			}
			lastMarker = index;
			return lastMarker;
		}
	
		public function release(marker:int):void {
			// no resources to release
		}
	
		public function get index():int {
			return p;
		}
	
		public function rewindTo(marker:int):void {
			seek(marker);
		}
	
		public function rewind():void {
			seek(lastMarker);
		}
	
		public function seek(index:int):void {
			if ( p==-1 ) {
				fillBuffer();
			}
			p = index;
		}
	
		/** Make stream jump to a new location, saving old location.
		 *  Switch back with pop().
		 */
		public function push(index:int):void {
			if ( calls==null ) {
				calls = new Array();
			}
			calls.push(p); // save current index
			seek(index);
		}
	
		/** Seek back to previous index saved during last push() call.
		 *  Return top of stack (return index).
		 */
		public function pop():int {
			var ret:int = calls.pop();
			seek(ret);
			return ret;
		}
	
		public function reset():void {
			p = 0;
			lastMarker = 0;
	        if (calls != null) {
	            calls = new Array();
	        }
	    }
	    
		public function get size():int {
			if ( p==-1 ) {
				fillBuffer();
			}
			return nodes.length;
		}
	
		// TREE REWRITE INTERFACE
		public function replaceChildren(parent:Object, startChildIndex:int, stopChildIndex:int, t:Object):void {
			if ( parent!=null ) {
				adaptor.replaceChildren(parent, startChildIndex, stopChildIndex, t);
			}
		}

		/** Used for testing, just return the token type stream */
		public function toString():String {
			if ( p==-1 ) {
				fillBuffer();
			}
			var buf:String = "";
			for (var i:int = 0; i < nodes.length; i++) {
				var t:Object = nodes[i];
				buf += " ";
				buf += (adaptor.getType(t));
			}
			return buf.toString();
		}
	
    	/** Debugging */
    	public function toTokenString(start:int, stop:int):String {
    		if ( p==-1 ) {
    			fillBuffer();
    		}
    		var buf:String = "";
    		for (var i:int = start; i < nodes.size() && i <= stop; i++) {
    			var t:Object = nodes[i];
    			buf += " ";
    			buf += adaptor.getToken(t);
    		}
    		return buf;
    	}
	
		public function toStringWithRange(start:Object, stop:Object):String {
			if ( start==null || stop==null ) {
				return null;
			}
			if ( p==-1 ) {
				fillBuffer();
			}
			trace("stop: "+stop);
			if ( start is CommonTree )
				trace("toString: "+CommonTree(start).token+", ");
			else
				trace(start);
			if ( stop is CommonTree )
				trace(CommonTree(stop).token);
			else
				trace(stop);
			// if we have the token stream, use that to dump text in order
			if ( tokens!=null ) {
				var beginTokenIndex:int = adaptor.getTokenStartIndex(start);
				var endTokenIndex:int = adaptor.getTokenStopIndex(stop);
				// if it's a tree, use start/stop index from start node
				// else use token range from start/stop nodes
				if ( adaptor.getType(stop)==TokenConstants.UP ) {
					endTokenIndex = adaptor.getTokenStopIndex(start);
				}
				else if ( adaptor.getType(stop)==TokenConstants.EOF ) {
					endTokenIndex = size-2; // don't use EOF
				}
				return tokens.toStringWithRange(beginTokenIndex, endTokenIndex);
			}
			// walk nodes looking for start
			var t:Object = null;
			var i:int = 0;
			for (; i < nodes.length; i++) {
				t = nodes[i];
				if ( t==start ) {
					break;
				}
			}
			// now walk until we see stop, filling string buffer with text
			 var buf:String = "";
			t = nodes[i];
			while ( t!=stop ) {
				var text:String = adaptor.getText(t);
				if ( text==null ) {
					text = " "+ adaptor.getType(t);
				}
				buf += text;
				i++;
				t = nodes[i];
			}
			// include stop node too
			text = adaptor.getText(stop);
			if ( text==null ) {
				text = " " + adaptor.getType(stop);
			}
			buf += text;
			return buf.toString();
		}
	}

}