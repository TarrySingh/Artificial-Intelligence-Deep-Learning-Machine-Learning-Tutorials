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

import org.antlr.tool.ErrorManager;
import org.antlr.tool.Grammar;
import org.antlr.tool.GrammarSemanticsMessage;
import org.antlr.tool.GrammarSyntaxMessage;
import org.antlr.Tool;
import org.antlr.codegen.CodeGenerator;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

/** Tree rewrites in tree parsers are basically identical to rewrites
 *  in a normal grammar except that the atomic element is a node not
 *  a Token.  Tests here ensure duplication of nodes occurs properly
 *  and basic functionality.
 */
public class TestTreeGrammarRewriteAST extends BaseTest {
	protected boolean debug = false;

	@Test public void testFlatList() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ID INT -> INT ID\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("34 abc\n", found);
	}

	@Test public void testSimpleTree() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(ID INT) -> ^(INT ID)\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("(34 abc)\n", found);
	}

	@Test public void testNonImaginaryWithCtor() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : INT ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : INT -> INT[\"99\"]\n" + // make new INT node
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
				    treeGrammar, "TP", "TLexer", "a", "a", "34");
		assertEquals("99\n", found);
	}

	@Test public void testCombinedRewriteAndAuto() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT) | INT ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(ID INT) -> ^(INT ID) | INT\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("(34 abc)\n", found);

		found = execTreeParser("T.g", grammar, "TParser", "TP.g",
							   treeGrammar, "TP", "TLexer", "a", "a", "34");
		assertEquals("34\n", found);
	}

	@Test public void testAvoidDup() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ID -> ^(ID ID)\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc");
		assertEquals("(abc abc)\n", found);
	}

	@Test public void testLoop() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID+ INT+ -> (^(ID INT))+ ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : (^(ID INT))+ -> INT+ ID+\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a b c 3 4 5");
		assertEquals("3 4 5 a b c\n", found);
	}

	@Test public void testAutoDup() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ID \n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc");
		assertEquals("abc\n", found);
	}

	@Test public void testAutoDupRule() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : b c ;\n" +
			"b : ID ;\n" +
			"c : INT ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 1");
		assertEquals("a 1\n", found);
	}

    @Test public void testAutoWildcard() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT ;\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "a : ID . \n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
        assertEquals("abc 34\n", found);
    }

    @Test public void testNoWildcardAsRootError() throws Exception {
        ErrorQueue equeue = new ErrorQueue();
        ErrorManager.setErrorListener(equeue);

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST;}\n" +
            "a : ^(. INT) \n" +
            "  ;\n";

        Grammar g = new Grammar(treeGrammar);
        Tool antlr = newTool();
        antlr.setOutputDirectory(null); // write to /dev/null
        CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
        g.setCodeGenerator(generator);
        generator.genRecognizer();

        assertEquals("unexpected errors: "+equeue, 1, equeue.errors.size());

        int expectedMsgID = ErrorManager.MSG_WILDCARD_AS_ROOT;
        Object expectedArg = null;
        antlr.RecognitionException expectedExc = null;
        GrammarSyntaxMessage expectedMessage =
            new GrammarSyntaxMessage(expectedMsgID, g, null, expectedArg, expectedExc);

        checkError(equeue, expectedMessage);        
    }

    @Test public void testAutoWildcard2() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT -> ^(ID INT);\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "a : ^(ID .) \n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
        assertEquals("(abc 34)\n", found);
    }

    @Test public void testAutoWildcardWithLabel() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT ;\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "a : ID c=. \n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
        assertEquals("abc 34\n", found);
    }

    @Test public void testAutoWildcardWithListLabel() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT ;\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "a : ID c+=. \n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
        assertEquals("abc 34\n", found);
    }

    @Test public void testAutoDupMultiple() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID ID INT;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ID ID INT\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a b 3");
		assertEquals("a b 3\n", found);
	}

	@Test public void testAutoDupTree() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(ID INT)\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 3");
		assertEquals("(a 3)\n", found);
	}

	@Test public void testAutoDupTree2() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT INT -> ^(ID INT INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(ID b b)\n" +
			"  ;\n" +
			"b : INT ;";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 3 4");
		assertEquals("(a 3 4)\n", found);
	}

	@Test public void testAutoDupTreeWithLabels() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(x=ID y=INT)\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 3");
		assertEquals("(a 3)\n", found);
	}

	@Test public void testAutoDupTreeWithListLabels() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(x+=ID y+=INT)\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 3");
		assertEquals("(a 3)\n", found);
	}

	@Test public void testAutoDupTreeWithRuleRoot() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(b INT) ;\n" +
			"b : ID ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 3");
		assertEquals("(a 3)\n", found);
	}

	@Test public void testAutoDupTreeWithRuleRootAndLabels() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(x=b INT) ;\n" +
			"b : ID ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 3");
		assertEquals("(a 3)\n", found);
	}

	@Test public void testAutoDupTreeWithRuleRootAndListLabels() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(x+=b y+=c) ;\n" +
			"b : ID ;\n" +
			"c : INT ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a 3");
		assertEquals("(a 3)\n", found);
	}

	@Test public void testAutoDupNestedTree() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : x=ID y=ID INT -> ^($x ^($y INT));\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(ID ^(ID INT))\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a b 3");
		assertEquals("(a (b 3))\n", found);
	}

	@Test public void testAutoDupTreeWithSubruleInside() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"tokens {OP;}\n" +
			"a : (x=ID|x=INT) -> ^(OP $x) ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(OP (b|c)) ;\n" +
			"b : ID ;\n" +
			"c : INT ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "a");
		assertEquals("(OP a)\n", found);
	}

	@Test public void testDelete() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ID -> \n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc");
		assertEquals("", found);
	}

	@Test public void testSetMatchNoRewrite() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : b INT\n" +
			"  ;\n" +
			"b : ID | INT ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("abc 34\n", found);
	}

	@Test public void testSetOptionalMatchNoRewrite() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : (ID|INT)? INT ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("abc 34\n", found);
	}


	@Test public void testSetMatchNoRewriteLevel2() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : x=ID INT -> ^($x INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^(ID (ID | INT) ) ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("(abc 34)\n", found);
	}

	@Test public void testSetMatchNoRewriteLevel2Root() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : x=ID INT -> ^($x INT);\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
			"a : ^((ID | INT) INT) ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("(abc 34)\n", found);
	}


	// REWRITE MODE

	@Test public void testRewriteModeCombinedRewriteAndAuto() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT) | INT ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"a : ^(ID INT) -> ^(ID[\"ick\"] INT)\n" +
			"  | INT\n" + // leaves it alone, returning $a.start
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "a", "abc 34");
		assertEquals("(ick 34)\n", found);

		found = execTreeParser("T.g", grammar, "TParser", "TP.g",
							   treeGrammar, "TP", "TLexer", "a", "a", "34");
		assertEquals("34\n", found);
	}

	@Test public void testRewriteModeFlatTree() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ID INT | INT ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"s : ID a ;\n" +
			"a : INT -> INT[\"1\"]\n"+
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
		assertEquals("abc 1\n", found);
	}

	@Test public void testRewriteModeChainRuleFlatTree() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ID INT | INT ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"s : a ;\n" +
			"a : b ;\n" +
			"b : ID INT -> INT ID\n"+
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
		assertEquals("34 abc\n", found);
	}

	@Test public void testRewriteModeChainRuleTree() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT) ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"s : a ;\n" +
			"a : b ;\n" + // a.tree must become b.tree
			"b : ^(ID INT) -> INT\n"+
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
		assertEquals("34\n", found);
	}

	@Test public void testRewriteModeChainRuleTree2() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID INT) ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"tokens { X; }\n" +
			"s : a* b ;\n" + // only b contributes to tree, but it's after a*; s.tree = b.tree
			"a : X ;\n" +
			"b : ^(ID INT) -> INT\n"+
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
		assertEquals("34\n", found);
	}

	@Test public void testRewriteModeChainRuleTree3() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : 'boo' ID INT -> 'boo' ^(ID INT) ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"tokens { X; }\n" +
			"s : 'boo' a* b ;\n" + // don't reset s.tree to b.tree due to 'boo'
			"a : X ;\n" +
			"b : ^(ID INT) -> INT\n"+
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "boo abc 34");
		assertEquals("boo 34\n", found);
	}

	@Test public void testRewriteModeChainRuleTree4() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : 'boo' ID INT -> ^('boo' ^(ID INT)) ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"tokens { X; }\n" +
			"s : ^('boo' a* b) ;\n" + // don't reset s.tree to b.tree due to 'boo'
			"a : X ;\n" +
			"b : ^(ID INT) -> INT\n"+
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "boo abc 34");
		assertEquals("(boo 34)\n", found);
	}

	@Test public void testRewriteModeChainRuleTree5() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : 'boo' ID INT -> ^('boo' ^(ID INT)) ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"tokens { X; }\n" +
			"s : ^(a b) ;\n" + // s.tree is a.tree
			"a : 'boo' ;\n" +
			"b : ^(ID INT) -> INT\n"+
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "boo abc 34");
		assertEquals("(boo 34)\n", found);
	}

    @Test public void testRewriteOfRuleRef() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT -> ID INT | INT ;\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : a -> a ;\n" +
            "a : ID INT -> ID INT ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
        assertEquals("abc 34\n", found);
    }

    @Test public void testRewriteOfRuleRefRoot() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT INT -> ^(INT ^(ID INT));\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : ^(a ^(ID INT)) -> a ;\n" +
            "a : INT ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 12 34");
        // emits whole tree when you ref the root since I can't know whether
        // you want the children or not.  You might be returning a whole new
        // tree.  Hmm...still seems weird.  oh well.
        assertEquals("(12 (abc 34))\n", found);
    }

    @Test public void testRewriteOfRuleRefRootLabeled() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT INT -> ^(INT ^(ID INT));\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : ^(label=a ^(ID INT)) -> a ;\n" +
            "a : INT ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 12 34");
        // emits whole tree when you ref the root since I can't know whether
        // you want the children or not.  You might be returning a whole new
        // tree.  Hmm...still seems weird.  oh well.
        assertEquals("(12 (abc 34))\n", found);
    }

    @Test public void testRewriteOfRuleRefRootListLabeled() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT INT -> ^(INT ^(ID INT));\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : ^(label+=a ^(ID INT)) -> a ;\n" +
            "a : INT ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 12 34");
        // emits whole tree when you ref the root since I can't know whether
        // you want the children or not.  You might be returning a whole new
        // tree.  Hmm...still seems weird.  oh well.
        assertEquals("(12 (abc 34))\n", found);
    }

    @Test public void testRewriteOfRuleRefChild() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT -> ^(ID ^(INT INT));\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : ^(ID a) -> a ;\n" +
            "a : ^(INT INT) ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
        assertEquals("(34 34)\n", found);
    }

    @Test public void testRewriteOfRuleRefLabel() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT -> ^(ID ^(INT INT));\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : ^(ID label=a) -> a ;\n" +
            "a : ^(INT INT) ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
        assertEquals("(34 34)\n", found);
    }

    @Test public void testRewriteOfRuleRefListLabel() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT -> ^(ID ^(INT INT));\n" +
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : ^(ID label+=a) -> a ;\n" +
            "a : ^(INT INT) ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
        assertEquals("(34 34)\n", found);
    }

    @Test public void testRewriteModeWithPredicatedRewrites() throws Exception {
		String grammar =
			"grammar T;\n" +
			"options {output=AST;}\n" +
			"a : ID INT -> ^(ID[\"root\"] ^(ID INT)) | INT -> ^(ID[\"root\"] INT) ;\n" +
			"ID : 'a'..'z'+ ;\n" +
			"INT : '0'..'9'+;\n" +
			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

		String treeGrammar =
			"tree grammar TP;\n"+
			"options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
			"s : ^(ID a) {System.out.println(\"altered tree=\"+$s.start.toStringTree());};\n" +
			"a : ^(ID INT) -> {true}? ^(ID[\"ick\"] INT)\n" +
			"              -> INT\n" +
			"  ;\n";

		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
									  treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
		assertEquals("altered tree=(root (ick 34))\n" +
					 "(root (ick 34))\n", found);
	}

    @Test public void testWildcardSingleNode() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT -> ^(ID[\"root\"] INT);\n"+
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "s : ^(ID c=.) -> $c\n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
        assertEquals("34\n", found);
    }

    @Test public void testWildcardUnlabeledSingleNode() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID INT -> ^(ID INT);\n"+
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "s : ^(ID .) -> ID\n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
        assertEquals("abc\n", found);
    }

    @Test public void testWildcardGrabsSubtree() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID x=INT y=INT z=INT -> ^(ID[\"root\"] ^($x $y $z));\n"+
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "s : ^(ID c=.) -> $c\n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 1 2 3");
        assertEquals("(1 2 3)\n", found);
    }

    @Test public void testWildcardGrabsSubtree2() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : ID x=INT y=INT z=INT -> ID ^($x $y $z);\n"+
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "s : ID c=. -> $c\n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "abc 1 2 3");
        assertEquals("(1 2 3)\n", found);
    }

    @Test public void testWildcardListLabel() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST;}\n" +
            "a : INT INT INT ;\n"+
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T;}\n" +
            "s : (c+=.)+ -> $c+\n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "1 2 3");
        assertEquals("1 2 3\n", found);
    }

    @Test public void testWildcardListLabel2() throws Exception {
        String grammar =
            "grammar T;\n" +
            "options {output=AST; ASTLabelType=CommonTree;}\n" +
            "a  : x=INT y=INT z=INT -> ^($x ^($y $z) ^($y $z));\n"+
            "ID : 'a'..'z'+ ;\n" +
            "INT : '0'..'9'+;\n" +
            "WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

        String treeGrammar =
            "tree grammar TP;\n"+
            "options {output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}\n" +
            "s : ^(INT (c+=.)+) -> $c+\n" +
            "  ;\n";

        String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
                                      treeGrammar, "TP", "TLexer", "a", "s", "1 2 3");
        assertEquals("(2 3) (2 3)\n", found);
    }
}
