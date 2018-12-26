package org.antlr.runtime {
	
    public class MissingTokenException extends MismatchedTokenException {
    	
    	public var inserted:Object;
    	
        public function MissingTokenException(expecting:int, input:IntStream, inserted:Object) {
            super(expecting, input);
            this.inserted = inserted;
        }
        
        public function get missingType():int {
            return expecting;
        }
        
        public override function toString():String {
            if ( inserted!=null && token!=null ) {
				return "MissingTokenException(inserted "+inserted+" at "+token.text+")";
			}
			if ( token!=null ) {
				return "MissingTokenException(at "+token.text+")";
			}
			return "MissingTokenException";
        }
        
    }
    
}