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
	import org.antlr.runtime.Token;
	import org.antlr.runtime.TokenConstants;
	
	/** A tree node that is wrapper for a Token object.  After 3.0 release
	 *  while building tree rewrite stuff, it became clear that computing
	 *  parent and child index is very difficult and cumbersome.  Better to
	 *  spend the space in every tree node.  If you don't want these extra
	 *  fields, it's easy to cut them out in your own BaseTree subclass.
	 */
	public class CommonTree extends BaseTree {
		/** A single token is the payload */
		protected var _token:Token;
		
		/** What token indexes bracket all tokens associated with this node
		 *  and below?
		 */
		public var startIndex:int=-1, stopIndex:int=-1;
	
		/** Who is the parent node of this node; if null, implies node is root */
		protected var _parent:CommonTree;
	
		/** What index is this node in the child list? Range: 0..n-1 */
		protected var _childIndex:int = -1;
			
		public function CommonTree(node:CommonTree = null) {
			if (node != null) {
				super(node);
				this._token = node._token;
				this.startIndex = node.startIndex;
				this.stopIndex = node.stopIndex;
			}
		}
	
		public static function createFromToken(t:Token):CommonTree {
			var ct:CommonTree = new CommonTree();
			ct._token = t;
			return ct;
		}
	
		public function get token():Token {
			return _token;
		}
	
		public override function dupNode():Tree {
			return new CommonTree(this);
		}
	
		public override function get isNil():Boolean {
			return _token==null;
		}
	
		public override function get type():int {
			if ( _token==null ) {
				return TokenConstants.INVALID_TOKEN_TYPE;
			}
			return _token.type;
		}
	
		public override function get text():String {
			if ( _token==null ) {
				return null;
			}
			return _token.text;
		}
	
		public override function get line():int {
			if ( _token==null || _token.line==0 ) {
				if ( childCount >0 ) {
					return getChild(0).line;
				}
				return 0;
			}
			return _token.line;
		}
	
		public override function get charPositionInLine():int {
			if ( _token==null || _token.charPositionInLine==-1 ) {
				if ( childCount>0 ) {
					return getChild(0).charPositionInLine;
				}
				return 0;
			}
			return _token.charPositionInLine;
		}
	
		public override function get tokenStartIndex():int {
			if ( startIndex==-1 && _token!=null ) {
				return _token.tokenIndex;
			}
			return startIndex;
		}
	
		public override function set tokenStartIndex(index:int):void {
			startIndex = index;
		}
	
		public override function get tokenStopIndex():int {
			if ( stopIndex==-1 && _token!=null ) {
				return _token.tokenIndex;
			}
			return stopIndex;
		}
	
		public override function set tokenStopIndex(index:int):void {
			stopIndex = index;
		}

		public override function get childIndex():int {
			return _childIndex;
		}
	
		public override function get parent():Tree {
			return _parent;
		}
	
		public override function set parent(t:Tree):void {
			this._parent = CommonTree(t);
		}
	
		public override function set childIndex(index:int):void {
			this._childIndex = index;
		}

		public function toString():String {
			if ( isNil ) {
				return "nil";
			}
			if ( type==TokenConstants.INVALID_TOKEN_TYPE ) {
    			return "<errornode>";
    		}
    		if ( token==null ) {
    			return null;
    		}
			return _token.text;
		}
	}

}
