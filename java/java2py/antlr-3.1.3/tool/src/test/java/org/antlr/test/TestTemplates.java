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

import org.antlr.tool.*;
import org.antlr.Tool;
import org.antlr.stringtemplate.StringTemplateGroup;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;
import org.antlr.codegen.CodeGenerator;
import org.antlr.grammar.v2.ANTLRParser;
import org.antlr.grammar.v3.ActionTranslator;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

/** Test templates in actions; %... shorthands */
public class TestTemplates extends BaseTest {
	private static final String LINE_SEP = System.getProperty("line.separator");

	@Test
    public void testTemplateConstructor() throws Exception {
		String action = "x = %foo(name={$ID.text});";
		String expecting = "x = templateLib.getInstanceOf(\"foo\"," +
			LINE_SEP + "  new STAttrMap().put(\"name\", (ID1!=null?ID1.getText():null)));";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
										"a",
										new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertNoErrors(equeue);

		assertEquals(expecting, found);
	}

	@Test
    public void testTemplateConstructorNoArgs() throws Exception {
		String action = "x = %foo();";
		String expecting = "x = templateLib.getInstanceOf(\"foo\");";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
										"a",
										new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertNoErrors(equeue);

		assertEquals(expecting, found);
	}

	@Test
    public void testIndirectTemplateConstructor() throws Exception {
		String action = "x = %({\"foo\"})(name={$ID.text});";
		String expecting = "x = templateLib.getInstanceOf(\"foo\"," +
			LINE_SEP + "  new STAttrMap().put(\"name\", (ID1!=null?ID1.getText():null)));";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
										"a",
										new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertNoErrors(equeue);

		assertEquals(expecting, found);
	}

	@Test public void testStringConstructor() throws Exception {
		String action = "x = %{$ID.text};";
		String expecting = "x = new StringTemplate(templateLib,(ID1!=null?ID1.getText():null));";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,
																	 "a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertNoErrors(equeue);

		assertEquals(expecting, found);
	}

	@Test public void testSetAttr() throws Exception {
		String action = "%x.y = z;";
		String expecting = "(x).setAttribute(\"y\", z);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
										"a",
										new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertNoErrors(equeue);

		assertEquals(expecting, found);
	}

	@Test public void testSetAttrOfExpr() throws Exception {
		String action = "%{foo($ID.text).getST()}.y = z;";
		String expecting = "(foo((ID1!=null?ID1.getText():null)).getST()).setAttribute(\"y\", z);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,
																	 "a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertNoErrors(equeue);

		assertEquals(expecting, found);
	}

	@Test public void testSetAttrOfExprInMembers() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"@members {\n" +
			"%code.instr = o;" + // must not get null ptr!
			"}\n" +
			"a : ID\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		assertNoErrors(equeue);
	}

	@Test public void testCannotHaveSpaceBeforeDot() throws Exception {
		String action = "%x .y = z;";
		String expecting = null;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		int expectedMsgID = ErrorManager.MSG_INVALID_TEMPLATE_ACTION;
		Object expectedArg = "%x";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testCannotHaveSpaceAfterDot() throws Exception {
		String action = "%x. y = z;";
		String expecting = null;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {\n" +
			"    output=template;\n" +
			"}\n" +
			"\n" +
			"a : ID {"+action+"}\n" +
			"  ;\n" +
			"\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		int expectedMsgID = ErrorManager.MSG_INVALID_TEMPLATE_ACTION;
		Object expectedArg = "%x.";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	protected void checkError(ErrorQueue equeue,
							  GrammarSemanticsMessage expectedMessage)
		throws Exception
	{
		/*
		System.out.println(equeue.infos);
		System.out.println(equeue.warnings);
		System.out.println(equeue.errors);
		*/
		Message foundMsg = null;
		for (int i = 0; i < equeue.errors.size(); i++) {
			Message m = (Message)equeue.errors.get(i);
			if (m.msgID==expectedMessage.msgID ) {
				foundMsg = m;
			}
		}
		assertTrue("no error; "+expectedMessage.msgID+" expected", equeue.errors.size()>0);
		assertTrue("too many errors; "+equeue.errors, equeue.errors.size()<=1);
		assertTrue("couldn't find expected error: "+expectedMessage.msgID, foundMsg!=null);
		assertTrue("error is not a GrammarSemanticsMessage",
				   foundMsg instanceof GrammarSemanticsMessage);
		assertEquals(expectedMessage.arg, foundMsg.arg);
		assertEquals(expectedMessage.arg2, foundMsg.arg2);
	}

	// S U P P O R T
	private void assertNoErrors(ErrorQueue equeue) {
		assertTrue("unexpected errors: "+equeue, equeue.errors.size()==0);
	}
}