package org.antlr.runtime
{
    public class UnwantedTokenException extends MismatchedTokenException
    {
        public function UnwantedTokenException(expecting:int, input:IntStream)
        {
            super(expecting, input);
        }
     
        public function get unexpectedToken():Token {
            return token;
        }
        
        public override function toString():String {
    		var exp:String = ", expected "+expecting;
			if ( expecting==TokenConstants.INVALID_TOKEN_TYPE ) {
				exp = "";
			}
			if ( token==null ) {
				return "UnwantedTokenException(found="+null+exp+")";
			}
			return "UnwantedTokenException(found="+token.text+exp+")";
        }
    }
}