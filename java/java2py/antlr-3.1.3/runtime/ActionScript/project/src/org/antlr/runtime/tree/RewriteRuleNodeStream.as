
package org.antlr.runtime.tree {
    
    /** Queues up nodes matched on left side of -> in a tree parser. This is
     *  the analog of RewriteRuleTokenStream for normal parsers. 
     */
    public class RewriteRuleNodeStream extends RewriteRuleElementStream {
        
        public function RewriteRuleNodeStream(adaptor:TreeAdaptor, elementDescription:String, element:Object = null) {
            super(adaptor, elementDescription, element);
        }
        
        public function nextNode():Object {
		    return _next();
    	}
    
    	protected override function toTree(el:Object):Object {
    		return adaptor.dupNode(el);
    	}
    
    	protected override function dup(el:Object):Object {
    		// we dup every node, so don't have to worry about calling dup; short-
    		// circuited next() so it doesn't call.
    		throw new Error("dup can't be called for a node stream.");
    	}
    }
}