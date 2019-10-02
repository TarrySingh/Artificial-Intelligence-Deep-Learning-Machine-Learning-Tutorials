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
    
    public class RewriteRuleSubtreeStream extends RewriteRuleElementStream {
        
        public function RewriteRuleSubtreeStream(adaptor:TreeAdaptor, elementDescription:String, element:Object=null) {
            super(adaptor, elementDescription, element);
        }
        
    	/** Treat next element as a single node even if it's a subtree.
    	 *  This is used instead of next() when the result has to be a
    	 *  tree root node.  Also prevents us from duplicating recently-added
    	 *  children; e.g., ^(type ID)+ adds ID to type and then 2nd iteration
    	 *  must dup the type node, but ID has been added.
    	 *
    	 *  Referencing a rule result twice is ok; dup entire tree as
    	 *  we can't be adding trees as root; e.g., expr expr.
    	 *
    	 *  Hideous code duplication here with super.next().  Can't think of
    	 *  a proper way to refactor.  This needs to always call dup node
    	 *  and super.next() doesn't know which to call: dup node or dup tree.
    	 */
    	public function nextNode():Object {
    		var n:int = size;
    		var el:Object;
    		if ( dirty || (cursor>=n && n==1) ) {
    			// if out of elements and size is 1, dup (at most a single node
    			// since this is for making root nodes).
    			el = _next();
    			return adaptor.dupNode(el);
    		}
    		// test size above then fetch
    		el = _next();
    		return el;
    	}
    
    	protected override function dup(el:Object):Object {
    		return adaptor.dupTree(el);
    	}
    }
}