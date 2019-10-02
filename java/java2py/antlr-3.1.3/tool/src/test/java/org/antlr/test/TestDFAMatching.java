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
package org.antlr.test;

import org.antlr.analysis.DFA;
import org.antlr.analysis.NFA;
import org.antlr.runtime.ANTLRStringStream;
import org.antlr.tool.Grammar;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestDFAMatching extends BaseTest {

    /** Public default constructor used by TestRig */
    public TestDFAMatching() {
    }

    @Test public void testSimpleAltCharTest() throws Exception {
        Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "A : {;}'a' | 'b' | 'c';");
		g.buildNFA();
		g.createLookaheadDFAs(false);
        DFA dfa = g.getLookaheadDFA(1);
        checkPrediction(dfa,"a",1);
        checkPrediction(dfa,"b",2);
        checkPrediction(dfa,"c",3);
        checkPrediction(dfa,"d", NFA.INVALID_ALT_NUMBER);
    }

    @Test public void testSets() throws Exception {
        Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "A : {;}'a'..'z' | ';' | '0'..'9' ;");
		g.buildNFA();
        g.createLookaheadDFAs(false);
        DFA dfa = g.getLookaheadDFA(1);
        checkPrediction(dfa,"a",1);
        checkPrediction(dfa,"q",1);
        checkPrediction(dfa,"z",1);
        checkPrediction(dfa,";",2);
        checkPrediction(dfa,"9",3);
    }

    @Test public void testFiniteCommonLeftPrefixes() throws Exception {
        Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "A : 'a' 'b' | 'a' 'c' | 'd' 'e' ;");
		g.buildNFA();
        g.createLookaheadDFAs(false);
        DFA dfa = g.getLookaheadDFA(1);
        checkPrediction(dfa,"ab",1);
        checkPrediction(dfa,"ac",2);
        checkPrediction(dfa,"de",3);
        checkPrediction(dfa,"q", NFA.INVALID_ALT_NUMBER);
    }

    @Test public void testSimpleLoops() throws Exception {
        Grammar g = new Grammar(
                "lexer grammar t;\n"+
                "A : (DIGIT)+ '.' DIGIT | (DIGIT)+ ;\n" +
                "fragment DIGIT : '0'..'9' ;\n");
		g.buildNFA();
        g.createLookaheadDFAs(false);
        DFA dfa = g.getLookaheadDFA(3);
        checkPrediction(dfa,"32",2);
        checkPrediction(dfa,"999.2",1);
        checkPrediction(dfa,".2", NFA.INVALID_ALT_NUMBER);
    }

    protected void checkPrediction(DFA dfa, String input, int expected)
        throws Exception
    {
        ANTLRStringStream stream = new ANTLRStringStream(input);
        assertEquals(dfa.predict(stream), expected);
    }

}
