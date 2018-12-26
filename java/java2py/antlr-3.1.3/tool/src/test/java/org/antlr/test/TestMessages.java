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
package org.antlr.test;

import org.antlr.Tool;
import org.antlr.codegen.CodeGenerator;
import org.antlr.grammar.v2.ANTLRParser;
import org.antlr.grammar.v3.ActionTranslator;
import org.antlr.tool.*;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestMessages extends BaseTest {

	/** Public default constructor used by TestRig */
	public TestMessages() {
	}


	@Test public void testMessageStringificationIsConsistent() throws Exception {
		String action = "$other.tree = null;";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"options { output = AST;}" +
			"otherrule\n" +
			"    : 'y' ;" +
			"rule\n" +
			"    : other=otherrule {" + action +"}\n" +
			"    ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,
																	"rule",
																	new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();

		int expectedMsgID = ErrorManager.MSG_WRITE_TO_READONLY_ATTR;
		Object expectedArg = "other";
		Object expectedArg2 = "tree";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		String expectedMessageString = expectedMessage.toString();
		assertEquals(expectedMessageString, expectedMessage.toString());
	}
}
