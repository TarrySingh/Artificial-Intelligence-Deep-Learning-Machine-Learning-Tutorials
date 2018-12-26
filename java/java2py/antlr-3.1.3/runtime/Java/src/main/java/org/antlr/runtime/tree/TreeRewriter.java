/*
 [The "BSD licence"]
 Copyright (c) 2005-2008 Terence Parr
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
package org.antlr.runtime.tree;

import org.antlr.runtime.RecognizerSharedState;
import org.antlr.runtime.RecognitionException;
import org.antlr.runtime.TokenStream;

public class TreeRewriter extends TreeParser {
    public interface fptr {
        public Object rule() throws RecognitionException;
    }

    protected TokenStream originalTokenStream;
    protected TreeAdaptor originalAdaptor;
    
    public TreeRewriter(TreeNodeStream input) {
        super(input);
    }
    public TreeRewriter(TreeNodeStream input, RecognizerSharedState state) {
        super(input, state);
        originalAdaptor = input.getTreeAdaptor();
        originalTokenStream = input.getTokenStream();        
    }

    public Object applyOnce(Object t, fptr whichRule) {
        if ( t==null ) return null;
        try {
            // share TreeParser object but not parsing-related state
            state = new RecognizerSharedState();
            input = new CommonTreeNodeStream(originalAdaptor, t);
            ((CommonTreeNodeStream)input).setTokenStream(originalTokenStream);
            setBacktrackingLevel(1);
            TreeRuleReturnScope r = (TreeRuleReturnScope)whichRule.rule();
            setBacktrackingLevel(0);
            if ( failed() ) return t;
            if ( r!=null && !t.equals(r.getTree()) && r.getTree()!=null ) { // show any transformations
                System.out.println(((CommonTree)t).toStringTree()+" -> "+
                                   ((CommonTree)r.getTree()).toStringTree());
            }
            if ( r!=null && r.getTree()!=null ) return r.getTree();
            else return t;
        }
        catch (RecognitionException e) { ; }
        return t;
    }

    public Object applyRepeatedly(Object t, fptr whichRule) {
        boolean treeChanged = true;
        while ( treeChanged ) {
            Object u = applyOnce(t, whichRule);
            treeChanged = !t.equals(u);
            t = u;
        }
        return t;
    }

    public Object downup(Object t) {
        TreeVisitor v = new TreeVisitor(new CommonTreeAdaptor());
        TreeVisitorAction actions = new TreeVisitorAction() {
            public Object pre(Object t)  { return applyOnce(t, topdown_fptr); }
            public Object post(Object t) { return applyRepeatedly(t, bottomup_ftpr); }
        };
        t = v.visit(t, actions);
        return t;
    }

    fptr topdown_fptr = new fptr() {
        public Object rule() throws RecognitionException { return topdown(); }
    };
    
    fptr bottomup_ftpr = new fptr() {
        public Object rule() throws RecognitionException { return bottomup(); }
    };

    // methods the downup strategy uses to do the up and down rules.
    // to override, just define tree grammar rule topdown and turn on
    // filter=true.
    public Object topdown() throws RecognitionException { return null; }
    public Object bottomup() throws RecognitionException { return null; }
}
