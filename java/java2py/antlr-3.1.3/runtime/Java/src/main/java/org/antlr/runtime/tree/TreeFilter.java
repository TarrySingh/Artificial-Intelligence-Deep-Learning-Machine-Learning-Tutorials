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

public class TreeFilter extends TreeParser {
    public interface fptr {
        public void rule() throws RecognitionException;
    }

    protected TokenStream originalTokenStream;
    protected TreeAdaptor originalAdaptor;

    public TreeFilter(TreeNodeStream input) {
        super(input);
    }
    public TreeFilter(TreeNodeStream input, RecognizerSharedState state) {
        super(input, state);
        originalAdaptor = input.getTreeAdaptor();
        originalTokenStream = input.getTokenStream();
    }

    public void applyOnce(Object t, fptr whichRule) {
        if ( t==null ) return;
        try {
            // share TreeParser object but not parsing-related state
            state = new RecognizerSharedState();
            input = new CommonTreeNodeStream(originalAdaptor, t);
            ((CommonTreeNodeStream)input).setTokenStream(originalTokenStream);
            setBacktrackingLevel(1);
            whichRule.rule();
            setBacktrackingLevel(0);
        }
        catch (RecognitionException e) { ; }
    }

    public void downup(Object t) {
        TreeVisitor v = new TreeVisitor(new CommonTreeAdaptor());
        TreeVisitorAction actions = new TreeVisitorAction() {
            public Object pre(Object t)  { applyOnce(t, topdown_fptr); return t; }
            public Object post(Object t) { applyOnce(t, bottomup_fptr); return t; }
        };
        v.visit(t, actions);
    }
        
    fptr topdown_fptr = new fptr() {
        public void rule() throws RecognitionException {
            topdown();
        }
    };

    fptr bottomup_fptr = new fptr() {
        public void rule() throws RecognitionException {
            bottomup();
        }
    };

    // methods the downup strategy uses to do the up and down rules.
    // to override, just define tree grammar rule topdown and turn on
    // filter=true.
    public void topdown() throws RecognitionException {;}
    public void bottomup() throws RecognitionException {;}
}
