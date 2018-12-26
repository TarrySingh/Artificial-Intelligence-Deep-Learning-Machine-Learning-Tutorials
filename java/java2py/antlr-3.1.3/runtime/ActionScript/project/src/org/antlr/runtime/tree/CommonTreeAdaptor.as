package org.antlr.runtime.tree {
	import org.antlr.runtime.CommonToken;
	import org.antlr.runtime.Token;
	import org.antlr.runtime.TokenConstants;
	
	/** A TreeAdaptor that works with any Tree implementation.  It provides
	 *  really just factory methods; all the work is done by BaseTreeAdaptor.
	 *  If you would like to have different tokens created than ClassicToken
	 *  objects, you need to override this and then set the parser tree adaptor to
	 *  use your subclass.
	 *
	 *  To get your parser to build nodes of a different type, override
     *  create(Token), errorNode(), and to be safe, YourTreeClass.dupNode().
     *  dupNode is called to duplicate nodes during rewrite operations.
	 */
	public class CommonTreeAdaptor extends BaseTreeAdaptor {
		/** Duplicate a node.  This is part of the factory;
		 *	override if you want another kind of node to be built.
		 *
		 *  I could use reflection to prevent having to override this
		 *  but reflection is slow.
		 */
		public override function dupNode(t:Object):Object {
			if ( t==null ) {
				return null;
			}
			return (Tree(t)).dupNode();
		}
	
		public override function createWithPayload(payload:Token):Object {
			return CommonTree.createFromToken(payload);
		}
	
		/** Tell me how to create a token for use with imaginary token nodes.
		 *  For example, there is probably no input symbol associated with imaginary
		 *  token DECL, but you need to create it as a payload or whatever for
		 *  the DECL node as in ^(DECL type ID).
		 *
		 *  If you care what the token payload objects' type is, you should
		 *  override this method and any other createToken variant.
		 */
		public override function createTokenFromType(tokenType:int, text:String):Token {
			return new CommonToken(tokenType, text);
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
		 */
		public override function createToken(fromToken:Token):Token {
			return CommonToken.cloneToken(fromToken);
		}
	
		/** Track start/stop token for subtree root created for a rule.
		 *  Only works with Tree nodes.  For rules that match nothing,
		 *  seems like this will yield start=i and stop=i-1 in a nil node.
		 *  Might be useful info so I'll not force to be i..i.
		 */
		public override function setTokenBoundaries(t:Object, startToken:Token, stopToken:Token):void {
			if ( t==null ) {
				return;
			}
			var start:int = 0;
			var stop:int = 0;
			if ( startToken!=null ) {
				start = startToken.tokenIndex;
			}
			if ( stopToken!=null ) {
				stop = stopToken.tokenIndex;
			}
			Tree(t).tokenStartIndex = start;
			Tree(t).tokenStopIndex = stop;
		}
	
		public override function getTokenStartIndex(t:Object):int {
			if ( t==null ) {
				return -1;
			}
			return Tree(t).tokenStartIndex;
		}
	
		public override function getTokenStopIndex(t:Object):int {
			if ( t==null ) {
				return -1;
			}
			return Tree(t).tokenStopIndex;
		}
	
		public override function getText(t:Object):String {
			if ( t==null ) {
				return null;
			}
			return Tree(t).text;
		}
	
	    public override function getType(t:Object):int {
			if ( t==null ) {
				return TokenConstants.INVALID_TOKEN_TYPE;
			}
			return Tree(t).type;;
		}
	
		/** What is the Token associated with this node?  If
		 *  you are not using CommonTree, then you must
		 *  override this in your own adaptor.
		 */
		public override function getToken(t:Object):Token {
			if ( t is CommonTree ) {
				return CommonTree(t).token;
			}
			return null; // no idea what to do
		}
	
		public override function getChild(t:Object, i:int):Object {
			if ( t==null ) {
				return null;
			}
	        return Tree(t).getChild(i);
	    }
	
	    public override function getChildCount(t:Object):int {
			if ( t==null ) {
				return 0;
			}
	        return Tree(t).childCount;
	    }

		public override function getParent(t:Object):Object {
		    if (t == null) {
		        return null;
		    }
			return Tree(t).parent;
		}
	
		public override function setParent(t:Object, parent:Object):void {
			if (t != null) {
			     Tree(t).parent = Tree(parent);
			}
		}
	
		public override function getChildIndex(t:Object):int {
		    if (t == null) {
		        return 0;
		    }
			return Tree(t).childIndex;
		}
	
		public override function setChildIndex(t:Object, index:int):void {
			if (t != null) {
			     Tree(t).childIndex = index;
			}
		}
	
		public override function replaceChildren(parent:Object, startChildIndex:int, stopChildIndex:int, t:Object):void {
			if ( parent!=null ) {
				Tree(parent).replaceChildren(startChildIndex, stopChildIndex, t);
			}
		}
		
	}
	
}