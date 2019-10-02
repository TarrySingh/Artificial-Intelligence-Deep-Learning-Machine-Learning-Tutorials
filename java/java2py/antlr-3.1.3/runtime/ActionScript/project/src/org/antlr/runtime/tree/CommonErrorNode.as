package org.antlr.runtime.tree
{
    
    import org.antlr.runtime.*;
    
    public class CommonErrorNode extends CommonTree {
    
    	public var input:IntStream;
    	public var start:Token;
    	public var stop:Token;
    	public var trappedException:RecognitionException;
    
    	public function CommonErrorNode(input:TokenStream, start:Token, stop:Token,
    						   e:RecognitionException)
    	{
    		//System.out.println("start: "+start+", stop: "+stop);
    		if ( stop==null ||
    			 (stop.tokenIndex < start.tokenIndex &&
    			  stop.type!=TokenConstants.EOF) )
    		{
    			// sometimes resync does not consume a token (when LT(1) is
    			// in follow set.  So, stop will be 1 to left to start. adjust.
    			// Also handle case where start is the first token and no token
    			// is consumed during recovery; LT(-1) will return null.
    			stop = start;
    		}
    		this.input = input;
    		this.start = start;
    		this.stop = stop;
    		this.trappedException = e;
    	}
    
    	public override function get isNil():Boolean {
    		return false;
    	}
    
    	public function getType():int {
    		return TokenConstants.INVALID_TOKEN_TYPE;
    	}
    
    	public function getText():String {
    		var badText:String = null;
    		if ( start is Token ) {
    			var i:int = Token(start).tokenIndex;
    			var j:int = Token(stop).tokenIndex;
    			if ( Token(stop).type == TokenConstants.EOF ) {
    				j = TokenStream(input).size;
    			}
    			badText = TokenStream(input).toStringWithRange(i, j);
    		}
    		else if ( start is Tree ) {
    			badText = TreeNodeStream(input).toStringWithRange(start, stop);
    		}
    		else {
    			// people should subclass if they alter the tree type so this
    			// next one is for sure correct.
    			badText = "<unknown>";
    		}
    		return badText;
    	}
    
    	public override function toString():String {
    		if ( trappedException is MissingTokenException ) {
    			return "<missing type: "+
    				   MissingTokenException(trappedException).missingType+
    				   ">";
    		}
    		else if ( trappedException is UnwantedTokenException ) {
    			return "<extraneous: "+
    				   UnwantedTokenException(trappedException).unexpectedToken+
    				   ", resync="+getText()+">";
    		}
    		else if ( trappedException is MismatchedTokenException ) {
    			return "<mismatched token: "+trappedException.token+", resync="+getText()+">";
    		}
    		else if ( trappedException is NoViableAltException ) {
    			return "<unexpected: "+trappedException.token+
    				   ", resync="+getText()+">";
    		}
    		return "<error: "+getText()+">";
    	}
       
    }
}