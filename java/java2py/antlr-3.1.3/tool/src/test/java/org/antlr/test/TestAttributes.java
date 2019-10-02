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

import org.antlr.Tool;
import org.antlr.grammar.v3.ActionTranslator;
import org.antlr.codegen.CodeGenerator;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.StringTemplateGroup;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;
import org.antlr.tool.*;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import org.antlr.grammar.v2.ANTLRParser;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import static org.junit.Assert.*;

/** Check the $x, $x.y attributes.  For checking the actual
 *  translation, assume the Java target.  This is still a great test
 *  for the semantics of the $x.y stuff regardless of the target.
 */
public class TestAttributes extends BaseTest {

	/** Public default constructor used by TestRig */
	public TestAttributes() {
	}

	@Test public void testEscapedLessThanInAction() throws Exception {
		Grammar g = new Grammar();
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		String action = "i<3; '<xmltag>'";
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),0);
		String expecting = action;
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, "<action>");
		actionST.setAttribute("action", rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);
	}

	@Test public void testEscaped$InAction() throws Exception {
		String action = "int \\$n; \"\\$in string\\$\"";
		String expecting = "int $n; \"$in string$\"";
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"@members {"+action+"}\n"+
			"a[User u, int i]\n" +
			"        : {"+action+"}\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "a",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),0);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);
	}

	@Test public void testArguments() throws Exception {
		String action = "$i; $i.x; $u; $u.x";
		String expecting = "i; i.x; u; u.x";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[User u, int i]\n" +
			"        : {"+action+"}\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testComplicatedArgParsing() throws Exception {
		String action = "x, (*a).foo(21,33), 3.2+1, '\\n', "+
						"\"a,oo\\nick\", {bl, \"fdkj\"eck}";
		String expecting = "x, (*a).foo(21,33), 3.2+1, '\\n', \"a,oo\\nick\", {bl, \"fdkj\"eck}";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// now check in actual grammar.
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[User u, int i]\n" +
			"        : A a["+action+"] B\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =	translator.translate();
		assertEquals(expecting, rawTranslation);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testBracketArgParsing() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// now check in actual grammar.
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[String[\\] ick, int i]\n" +
			"        : A \n"+
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		Rule r = g.getRule("a");
		AttributeScope parameters = r.parameterScope;
		List<Attribute> attrs = parameters.getAttributes();
		assertEquals("attribute mismatch","String[] ick",attrs.get(0).decl.toString());
		assertEquals("parameter name mismatch","ick",attrs.get(0).name);
		assertEquals("declarator mismatch", "String[]", attrs.get(0).type);

		assertEquals("attribute mismatch","int i",attrs.get(1).decl.toString());
		assertEquals("parameter name mismatch","i",attrs.get(1).name);
		assertEquals("declarator mismatch", "int", attrs.get(1).type);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testStringArgParsing() throws Exception {
		String action = "34, '{', \"it's<\", '\"', \"\\\"\", 19";
		String expecting = "34, '{', \"it's<\", '\"', \"\\\"\", 19";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// now check in actual grammar.
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[User u, int i]\n" +
			"        : A a["+action+"] B\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =	translator.translate();
		assertEquals(expecting, rawTranslation);

		List<String> expectArgs = new ArrayList<String>() {
			{add("34");}
			{add("'{'");}
			{add("\"it's<\"");}
			{add("'\"'");}
			{add("\"\\\"\"");} // that's "\""
			{add("19");}
		};
		List<String> actualArgs = CodeGenerator.getListOfArgumentsFromAction(action, ',');
		assertEquals("args mismatch", expectArgs, actualArgs);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testComplicatedSingleArgParsing() throws Exception {
		String action = "(*a).foo(21,33,\",\")";
		String expecting = "(*a).foo(21,33,\",\")";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// now check in actual grammar.
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[User u, int i]\n" +
			"        : A a["+action+"] B\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =	translator.translate();
		assertEquals(expecting, rawTranslation);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testArgWithLT() throws Exception {
		String action = "34<50";
		String expecting = "34<50";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// now check in actual grammar.
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[boolean b]\n" +
			"        : A a["+action+"] B\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		assertEquals(expecting, rawTranslation);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testGenericsAsArgumentDefinition() throws Exception {
		String action = "$foo.get(\"ick\");";
		String expecting = "foo.get(\"ick\");";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		String grammar =
			"parser grammar T;\n"+
			"a[HashMap<String,String> foo]\n" +
			"        : {"+action+"}\n" +
			"        ;";
		Grammar g = new Grammar(grammar);
		Rule ra = g.getRule("a");
		List<Attribute> attrs = ra.parameterScope.getAttributes();
		assertEquals("attribute mismatch","HashMap<String,String> foo",attrs.get(0).decl.toString());
		assertEquals("parameter name mismatch","foo",attrs.get(0).name);
		assertEquals("declarator mismatch", "HashMap<String,String>", attrs.get(0).type);

		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testGenericsAsArgumentDefinition2() throws Exception {
		String action = "$foo.get(\"ick\"); x=3;";
		String expecting = "foo.get(\"ick\"); x=3;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		String grammar =
			"parser grammar T;\n"+
			"a[HashMap<String,String> foo, int x, List<String> duh]\n" +
			"        : {"+action+"}\n" +
			"        ;";
		Grammar g = new Grammar(grammar);
		Rule ra = g.getRule("a");
		List<Attribute> attrs = ra.parameterScope.getAttributes();

		assertEquals("attribute mismatch","HashMap<String,String> foo",attrs.get(0).decl.toString().trim());
		assertEquals("parameter name mismatch","foo",attrs.get(0).name);
		assertEquals("declarator mismatch", "HashMap<String,String>", attrs.get(0).type);

		assertEquals("attribute mismatch","int x",attrs.get(1).decl.toString().trim());
		assertEquals("parameter name mismatch","x",attrs.get(1).name);
		assertEquals("declarator mismatch", "int", attrs.get(1).type);

		assertEquals("attribute mismatch","List<String> duh",attrs.get(2).decl.toString().trim());
		assertEquals("parameter name mismatch","duh",attrs.get(2).name);
		assertEquals("declarator mismatch", "List<String>", attrs.get(2).type);

		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testGenericsAsReturnValue() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		String grammar =
			"parser grammar T;\n"+
			"a returns [HashMap<String,String> foo] : ;\n";
		Grammar g = new Grammar(grammar);
		Rule ra = g.getRule("a");
		List<Attribute> attrs = ra.returnScope.getAttributes();
		assertEquals("attribute mismatch","HashMap<String,String> foo",attrs.get(0).decl.toString());
		assertEquals("parameter name mismatch","foo",attrs.get(0).name);
		assertEquals("declarator mismatch", "HashMap<String,String>", attrs.get(0).type);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testComplicatedArgParsingWithTranslation() throws Exception {
		String action = "x, $A.text+\"3242\", (*$A).foo(21,33), 3.2+1, '\\n', "+
						"\"a,oo\\nick\", {bl, \"fdkj\"eck}";
		String expecting = "x, (A1!=null?A1.getText():null)+\"3242\", (*A1).foo(21,33), 3.2+1, '\\n', \"a,oo\\nick\", {bl, \"fdkj\"eck}";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);

		// now check in actual grammar.
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[User u, int i]\n" +
			"        : A a["+action+"] B\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	/** $x.start refs are checked during translation not before so ANTLR misses
	 the fact that rule r has refs to predefined attributes if the ref is after
	 the def of the method or self-referential.  Actually would be ok if I didn't
	 convert actions to strings; keep as templates.
	 June 9, 2006: made action translation leave templates not strings
	 */
	@Test public void testRefToReturnValueBeforeRefToPredefinedAttr() throws Exception {
		String action = "$x.foo";
		String expecting = "(x!=null?x.foo:0)";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : x=b {"+action+"} ;\n" +
			"b returns [int foo] : B {$b.start} ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRuleLabelBeforeRefToPredefinedAttr() throws Exception {
		// As of Mar 2007, I'm removing unused labels.  Unfortunately,
		// the action is not seen until code gen.  Can't see $x.text
		// before stripping unused labels.  We really need to translate
		// actions first so code gen logic can use info.
		String action = "$x.text";
		String expecting = "(x!=null?input.toString(x.start,x.stop):null)";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : x=b {"+action+"} ;\n" +
			"b : B ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testInvalidArguments() throws Exception {
		String action = "$x";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[User u, int i]\n" +
			"        : {"+action+"}\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator,
																	 "a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_SIMPLE_ATTRIBUTE;
		Object expectedArg = "x";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testReturnValue() throws Exception {
		String action = "$x.i";
		String expecting = "x";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a returns [int i]\n" +
			"        : 'a'\n" +
			"        ;\n" +
			"b : x=a {"+action+"} ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "b",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testReturnValueWithNumber() throws Exception {
		String action = "$x.i1";
		String expecting = "x";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a returns [int i1]\n" +
			"        : 'a'\n" +
			"        ;\n" +
			"b : x=a {"+action+"} ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "b",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testReturnValues() throws Exception {
		String action = "$i; $i.x; $u; $u.x";
		String expecting = "retval.i; retval.i.x; retval.u; retval.u.x";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a returns [User u, int i]\n" +
			"        : {"+action+"}\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	/* regression test for ANTLR-46 */
	@Test public void testReturnWithMultipleRuleRefs() throws Exception {
		String action1 = "$obj = $rule2.obj;";
		String action2 = "$obj = $rule3.obj;";
		String expecting1 = "obj = rule21;";
		String expecting2 = "obj = rule32;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"rule1 returns [ Object obj ]\n" +
			":	rule2 { "+action1+" }\n" +
			"|	rule3 { "+action2+" }\n" +
			";\n"+
			"rule2 returns [ Object obj ]\n"+
			":	foo='foo' { $obj = $foo.text; }\n"+
			";\n"+
			"rule3 returns [ Object obj ]\n"+
			":	bar='bar' { $obj = $bar.text; }\n"+
			";");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		int i = 0;
		String action = action1;
		String expecting = expecting1;
		do {
			ActionTranslator translator = new ActionTranslator(generator,"rule1",
																		 new antlr.CommonToken(ANTLRParser.ACTION,action),i+1);
			String rawTranslation =
					translator.translate();
			StringTemplateGroup templates =
					new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
			StringTemplate actionST = new StringTemplate(templates, rawTranslation);
			String found = actionST.toString();
			assertEquals(expecting, found);
			action = action2;
			expecting = expecting2;
		} while (i++ < 1);
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testInvalidReturnValues() throws Exception {
		String action = "$x";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a returns [User u, int i]\n" +
			"        : {"+action+"}\n" +
			"        ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_SIMPLE_ATTRIBUTE;
		Object expectedArg = "x";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testTokenLabels() throws Exception {
		String action = "$id; $f; $id.text; $id.getText(); $id.dork " +
						"$id.type; $id.line; $id.pos; " +
						"$id.channel; $id.index;";
		String expecting = "id; f; (id!=null?id.getText():null); id.getText(); id.dork (id!=null?id.getType():0); (id!=null?id.getLine():0); (id!=null?id.getCharPositionInLine():0); (id!=null?id.getChannel():0); (id!=null?id.getTokenIndex():0);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : id=ID f=FLOAT {"+action+"}\n" +
			"  ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRuleLabels() throws Exception {
		String action = "$r.x; $r.start;\n $r.stop;\n $r.tree; $a.x; $a.stop;";
		String expecting = "(r!=null?r.x:0); (r!=null?((Token)r.start):null);\n" +
						   "             (r!=null?((Token)r.stop):null);\n" +
						   "             (r!=null?((Object)r.tree):null); (r!=null?r.x:0); (r!=null?((Token)r.stop):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a returns [int x]\n" +
			"  :\n" +
			"  ;\n"+
			"b : r=a {###"+action+"!!!}\n" +
			"  ;");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // codegen phase sets some vars we need
		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testAmbiguRuleRef() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : A a {$a.text} | B ;");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		// error(132): <string>:2:9: reference $a is ambiguous; rule a is enclosing rule and referenced in the production
		assertEquals("unexpected errors: "+equeue, 1, equeue.errors.size());
	}

	@Test public void testRuleLabelsWithSpecialToken() throws Exception {
		String action = "$r.x; $r.start; $r.stop; $r.tree; $a.x; $a.stop;";
		String expecting = "(r!=null?r.x:0); (r!=null?((MYTOKEN)r.start):null); (r!=null?((MYTOKEN)r.stop):null); (r!=null?((Object)r.tree):null); (r!=null?r.x:0); (r!=null?((MYTOKEN)r.stop):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"options {TokenLabelType=MYTOKEN;}\n"+
			"a returns [int x]\n" +
			"  :\n" +
			"  ;\n"+
			"b : r=a {###"+action+"!!!}\n" +
			"  ;");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // codegen phase sets some vars we need

		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testForwardRefRuleLabels() throws Exception {
		String action = "$r.x; $r.start; $r.stop; $r.tree; $a.x; $a.tree;";
		String expecting = "(r!=null?r.x:0); (r!=null?((Token)r.start):null); (r!=null?((Token)r.stop):null); (r!=null?((Object)r.tree):null); (r!=null?r.x:0); (r!=null?((Object)r.tree):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"b : r=a {###"+action+"!!!}\n" +
			"  ;\n" +
			"a returns [int x]\n" +
			"  : ;\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // codegen phase sets some vars we need

		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testInvalidRuleLabelAccessesParameter() throws Exception {
		String action = "$r.z";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[int z] returns [int x]\n" +
			"  :\n" +
			"  ;\n"+
			"b : r=a[3] {"+action+"}\n" +
			"  ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_INVALID_RULE_PARAMETER_REF;
		Object expectedArg = "a";
		Object expectedArg2 = "z";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testInvalidRuleLabelAccessesScopeAttribute() throws Exception {
		String action = "$r.n";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a\n" +
			"scope { int n; }\n" +
			"  :\n" +
			"  ;\n"+
			"b : r=a[3] {"+action+"}\n" +
			"  ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_INVALID_RULE_SCOPE_ATTRIBUTE_REF;
		Object expectedArg = "a";
		Object expectedArg2 = "n";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testInvalidRuleAttribute() throws Exception {
		String action = "$r.blort";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[int z] returns [int x]\n" +
			"  :\n" +
			"  ;\n"+
			"b : r=a[3] {"+action+"}\n" +
			"  ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_RULE_ATTRIBUTE;
		Object expectedArg = "a";
		Object expectedArg2 = "blort";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testMissingRuleAttribute() throws Exception {
		String action = "$r";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a[int z] returns [int x]\n" +
			"  :\n" +
			"  ;\n"+
			"b : r=a[3] {"+action+"}\n" +
			"  ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();

		int expectedMsgID = ErrorManager.MSG_ISOLATED_RULE_SCOPE;
		Object expectedArg = "r";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testMissingUnlabeledRuleAttribute() throws Exception {
		String action = "$a";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a returns [int x]:\n" +
			"  ;\n"+
			"b : a {"+action+"}\n" +
			"  ;");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();

		int expectedMsgID = ErrorManager.MSG_ISOLATED_RULE_SCOPE;
		Object expectedArg = "a";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testNonDynamicAttributeOutsideRule() throws Exception {
		String action = "public void foo() { $x; }";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"@members {'+action+'}\n" +
			"a : ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator,
																	 null,
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),0);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_ATTRIBUTE_REF_NOT_IN_RULE;
		Object expectedArg = "x";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testNonDynamicAttributeOutsideRule2() throws Exception {
		String action = "public void foo() { $x.y; }";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"@members {'+action+'}\n" +
			"a : ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator = new ActionTranslator(generator,
																	 null,
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),0);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_ATTRIBUTE_REF_NOT_IN_RULE;
		Object expectedArg = "x";
		Object expectedArg2 = "y";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	// D Y N A M I C A L L Y  S C O P E D  A T T R I B U T E S

	@Test public void testBasicGlobalScope() throws Exception {
		String action = "$Symbols::names.add($id.text);";
		String expecting = "((Symbols_scope)Symbols_stack.peek()).names.add((id!=null?id.getText():null));";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  int n;\n" +
			"  List names;\n" +
			"}\n" +
			"a scope Symbols; : (id=ID ';' {"+action+"} )+\n" +
			"  ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testUnknownGlobalScope() throws Exception {
		String action = "$Symbols::names.add($id.text);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a scope Symbols; : (id=ID ';' {"+action+"} )+\n" +
			"  ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);

		assertEquals("unexpected errors: "+equeue, 2, equeue.errors.size());

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_DYNAMIC_SCOPE;
		Object expectedArg = "Symbols";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testIndexedGlobalScope() throws Exception {
		String action = "$Symbols[-1]::names.add($id.text);";
		String expecting =
			"((Symbols_scope)Symbols_stack.elementAt(Symbols_stack.size()-1-1)).names.add((id!=null?id.getText():null));";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  int n;\n" +
			"  List names;\n" +
			"}\n" +
			"a scope Symbols; : (id=ID ';' {"+action+"} )+\n" +
			"  ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void test0IndexedGlobalScope() throws Exception {
		String action = "$Symbols[0]::names.add($id.text);";
		String expecting =
			"((Symbols_scope)Symbols_stack.elementAt(0)).names.add((id!=null?id.getText():null));";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  int n;\n" +
			"  List names;\n" +
			"}\n" +
			"a scope Symbols; : (id=ID ';' {"+action+"} )+\n" +
			"  ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		assertEquals(expecting, rawTranslation);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testAbsoluteIndexedGlobalScope() throws Exception {
		String action = "$Symbols[3]::names.add($id.text);";
		String expecting =
			"((Symbols_scope)Symbols_stack.elementAt(3)).names.add((id!=null?id.getText():null));";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  int n;\n" +
			"  List names;\n" +
			"}\n" +
			"a scope Symbols; : (id=ID ';' {"+action+"} )+\n" +
			"  ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		assertEquals(expecting, rawTranslation);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testScopeAndAttributeWithUnderscore() throws Exception {
		String action = "$foo_bar::a_b;";
		String expecting = "((foo_bar_scope)foo_bar_stack.peek()).a_b;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope foo_bar {\n" +
			"  int a_b;\n" +
			"}\n" +
			"a scope foo_bar; : (ID {"+action+"} )+\n" +
			"  ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testSharedGlobalScope() throws Exception {
		String action = "$Symbols::x;";
		String expecting = "((Symbols_scope)Symbols_stack.peek()).x;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  String x;\n" +
			"}\n" +
			"a\n"+
			"scope { int y; }\n"+
			"scope Symbols;\n" +
			" : b {"+action+"}\n" +
			" ;\n" +
			"b : ID {$Symbols::x=$ID.text} ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testGlobalScopeOutsideRule() throws Exception {
		String action = "public void foo() {$Symbols::names.add('foo');}";
		String expecting = "public void foo() {((Symbols_scope)Symbols_stack.peek()).names.add('foo');}";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  int n;\n" +
			"  List names;\n" +
			"}\n" +
			"@members {'+action+'}\n" +
			"a : \n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRuleScopeOutsideRule() throws Exception {
		String action = "public void foo() {$a::name;}";
		String expecting = "public void foo() {((a_scope)a_stack.peek()).name;}";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"@members {"+action+"}\n" +
			"a\n" +
			"scope { String name; }\n" +
			"  : {foo();}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,
																	 null,
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),0);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testBasicRuleScope() throws Exception {
		String action = "$a::n;";
		String expecting = "((a_scope)a_stack.peek()).n;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testUnqualifiedRuleScopeAccessInsideRule() throws Exception {
		String action = "$n;";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		int expectedMsgID = ErrorManager.MSG_ISOLATED_RULE_ATTRIBUTE;
		Object expectedArg = "n";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg,
										expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testIsolatedDynamicRuleScopeRef() throws Exception {
		String action = "$a;"; // refers to stack not top of stack
		String expecting = "a_stack;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : b ;\n" +
			"b : {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testDynamicRuleScopeRefInSubrule() throws Exception {
		String action = "$a::n;";
		String expecting = "((a_scope)a_stack.peek()).n;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a\n" +
			"scope {\n" +
			"  float n;\n" +
			"} : b ;\n" +
			"b : {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testIsolatedGlobalScopeRef() throws Exception {
		String action = "$Symbols;";
		String expecting = "Symbols_stack;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  String x;\n" +
			"}\n" +
			"a\n"+
			"scope { int y; }\n"+
			"scope Symbols;\n" +
			" : b {"+action+"}\n" +
			" ;\n" +
			"b : ID {$Symbols::x=$ID.text} ;\n" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRuleScopeFromAnotherRule() throws Exception {
		String action = "$a::n;"; // must be qualified
		String expecting = "((a_scope)a_stack.peek()).n;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a\n" +
			"scope {\n" +
			"  boolean n;\n" +
			"} : b\n" +
			"  ;\n" +
			"b : {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testFullyQualifiedRefToCurrentRuleParameter() throws Exception {
		String action = "$a.i;";
		String expecting = "i;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a[int i]: {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testFullyQualifiedRefToCurrentRuleRetVal() throws Exception {
		String action = "$a.i;";
		String expecting = "retval.i;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a returns [int i, int j]: {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testSetFullyQualifiedRefToCurrentRuleRetVal() throws Exception {
		String action = "$a.i = 1;";
		String expecting = "retval.i = 1;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a returns [int i, int j]: {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testIsolatedRefToCurrentRule() throws Exception {
		String action = "$a;";
		String expecting = "";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : 'a' {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		int expectedMsgID = ErrorManager.MSG_ISOLATED_RULE_SCOPE;
		Object expectedArg = "a";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg,
										expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testIsolatedRefToRule() throws Exception {
		String action = "$x;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : x=b {"+action+"}\n" +
			"  ;\n" +
			"b : 'b' ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		int expectedMsgID = ErrorManager.MSG_ISOLATED_RULE_SCOPE;
		Object expectedArg = "x";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	/*  I think these have to be errors $a.x makes no sense.
	@Test public void testFullyQualifiedRefToLabelInCurrentRule() throws Exception {
			String action = "$a.x;";
			String expecting = "x;";

			ErrorQueue equeue = new ErrorQueue();
			ErrorManager.setErrorListener(equeue);
			Grammar g = new Grammar(
				"grammar t;\n"+
					"a : x='a' {"+action+"}\n" +
					"  ;\n");
			Tool antlr = newTool();
			CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
			g.setCodeGenerator(generator);
			generator.genRecognizer(); // forces load of templates
			ActionTranslator translator = new ActionTranslator(generator,"a",
															   new antlr.CommonToken(ANTLRParser.ACTION,action),1);
			String rawTranslation =
				translator.translate();
			StringTemplateGroup templates =
				new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
			StringTemplate actionST = new StringTemplate(templates, rawTranslation);
			String found = actionST.toString();
			assertEquals(expecting, found);

			assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
		}

	@Test public void testFullyQualifiedRefToListLabelInCurrentRule() throws Exception {
		String action = "$a.x;"; // must be qualified
		String expecting = "list_x;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
				"a : x+='a' {"+action+"}\n" +
				"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
														   new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}
*/
	@Test public void testFullyQualifiedRefToTemplateAttributeInCurrentRule() throws Exception {
		String action = "$a.st;"; // can be qualified
		String expecting = "retval.st;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"options {output=template;}\n"+
			"a : (A->{$A.text}) {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRuleRefWhenRuleHasScope() throws Exception {
		String action = "$b.start;";
		String expecting = "(b1!=null?((Token)b1.start):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"a : b {###"+action+"!!!} ;\n" +
			"b\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : 'b' \n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testDynamicScopeRefOkEvenThoughRuleRefExists() throws Exception {
		String action = "$b::n;";
		String expecting = "((b_scope)b_stack.peek()).n;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"s : b ;\n"+
			"b\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : '(' b ')' {"+action+"}\n" + // refers to current invocation's n
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRefToTemplateAttributeForCurrentRule() throws Exception {
		String action = "$st=null;";
		String expecting = "retval.st =null;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"options {output=template;}\n"+
			"a : {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRefToTextAttributeForCurrentRule() throws Exception {
		String action = "$text";
		String expecting = "input.toString(retval.start,input.LT(-1))";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"options {output=template;}\n"+
			"a : {"+action+"}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRefToStartAttributeForCurrentRule() throws Exception {
		String action = "$start;";
		String expecting = "((Token)retval.start);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n" +
			"a : {###"+action+"!!!}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testTokenLabelFromMultipleAlts() throws Exception {
		String action = "$ID.text;"; // must be qualified
		String action2 = "$INT.text;"; // must be qualified
		String expecting = "(ID1!=null?ID1.getText():null);";
		String expecting2 = "(INT2!=null?INT2.getText():null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ID {"+action+"}\n" +
			"  | INT {"+action2+"}\n" +
			"  ;\n" +
			"ID : 'a';\n" +
			"INT : '0';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
		translator = new ActionTranslator(generator,
											   "a",
											   new antlr.CommonToken(ANTLRParser.ACTION,action2),2);
		rawTranslation =
			translator.translate();
		templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		actionST = new StringTemplate(templates, rawTranslation);
		found = actionST.toString();

		assertEquals(expecting2, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRuleLabelFromMultipleAlts() throws Exception {
		String action = "$b.text;"; // must be qualified
		String action2 = "$c.text;"; // must be qualified
		String expecting = "(b1!=null?input.toString(b1.start,b1.stop):null);";
		String expecting2 = "(c2!=null?input.toString(c2.start,c2.stop):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : b {"+action+"}\n" +
			"  | c {"+action2+"}\n" +
			"  ;\n" +
			"b : 'a';\n" +
			"c : '0';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
		translator = new ActionTranslator(generator,
											   "a",
											   new antlr.CommonToken(ANTLRParser.ACTION,action2),2);
		rawTranslation =
			translator.translate();
		templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		actionST = new StringTemplate(templates, rawTranslation);
		found = actionST.toString();

		assertEquals(expecting2, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testUnknownDynamicAttribute() throws Exception {
		String action = "$a::x";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : {"+action+"}\n" +
			"  ;\n");
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
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_DYNAMIC_SCOPE_ATTRIBUTE;
		Object expectedArg = "a";
		Object expectedArg2 = "x";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testUnknownGlobalDynamicAttribute() throws Exception {
		String action = "$Symbols::x";
		String expecting = action;

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"scope Symbols {\n" +
			"  int n;\n" +
			"}\n" +
			"a : {'+action+'}\n" +
			"  ;\n");
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
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_DYNAMIC_SCOPE_ATTRIBUTE;
		Object expectedArg = "Symbols";
		Object expectedArg2 = "x";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testUnqualifiedRuleScopeAttribute() throws Exception {
		String action = "$n;"; // must be qualified
		String expecting = "$n;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : b\n" +
			"  ;\n" +
			"b : {'+action+'}\n" +
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "b",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_SIMPLE_ATTRIBUTE;
		Object expectedArg = "n";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testRuleAndTokenLabelTypeMismatch() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : id='foo' id=b\n" +
			"  ;\n" +
			"b : ;\n");
		int expectedMsgID = ErrorManager.MSG_LABEL_TYPE_CONFLICT;
		Object expectedArg = "id";
		Object expectedArg2 = "rule!=token";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testListAndTokenLabelTypeMismatch() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ids+='a' ids='b'\n" +
			"  ;\n" +
			"b : ;\n");
		int expectedMsgID = ErrorManager.MSG_LABEL_TYPE_CONFLICT;
		Object expectedArg = "ids";
		Object expectedArg2 = "token!=token-list";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testListAndRuleLabelTypeMismatch() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"options {output=AST;}\n"+
			"a : bs+=b bs=b\n" +
			"  ;\n" +
			"b : 'b';\n");
		int expectedMsgID = ErrorManager.MSG_LABEL_TYPE_CONFLICT;
		Object expectedArg = "bs";
		Object expectedArg2 = "rule!=rule-list";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testArgReturnValueMismatch() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a[int i] returns [int x, int i]\n" +
			"  : \n" +
			"  ;\n" +
			"b : ;\n");
		int expectedMsgID = ErrorManager.MSG_ARG_RETVAL_CONFLICT;
		Object expectedArg = "i";
		Object expectedArg2 = "a";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testSimplePlusEqualLabel() throws Exception {
		String action = "$ids.size();"; // must be qualified
		String expecting = "list_ids.size();";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"parser grammar t;\n"+
			"a : ids+=ID ( COMMA ids+=ID {"+action+"})* ;\n");
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
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testPlusEqualStringLabel() throws Exception {
		String action = "$ids.size();"; // must be qualified
		String expecting = "list_ids.size();";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ids+='if' ( ',' ids+=ID {"+action+"})* ;" +
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
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testPlusEqualSetLabel() throws Exception {
		String action = "$ids.size();"; // must be qualified
		String expecting = "list_ids.size();";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ids+=('a'|'b') ( ',' ids+=ID {"+action+"})* ;" +
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
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testPlusEqualWildcardLabel() throws Exception {
		String action = "$ids.size();"; // must be qualified
		String expecting = "list_ids.size();";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ids+=. ( ',' ids+=ID {"+action+"})* ;" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "a",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testImplicitTokenLabel() throws Exception {
		String action = "$ID; $ID.text; $ID.getText()";
		String expecting = "ID1; (ID1!=null?ID1.getText():null); ID1.getText()";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ID {"+action+"} ;" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");

		ActionTranslator translator =
			new ActionTranslator(generator,
									  "a",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testImplicitRuleLabel() throws Exception {
		String action = "$r.start;";
		String expecting = "(r1!=null?((Token)r1.start):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : r {###"+action+"!!!} ;" +
			"r : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testReuseExistingLabelWithImplicitRuleLabel() throws Exception {
		String action = "$r.start;";
		String expecting = "(x!=null?((Token)x.start):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : x=r {###"+action+"!!!} ;" +
			"r : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testReuseExistingListLabelWithImplicitRuleLabel() throws Exception {
		String action = "$r.start;";
		String expecting = "(x!=null?((Token)x.start):null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"options {output=AST;}\n" +
			"a : x+=r {###"+action+"!!!} ;" +
			"r : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testReuseExistingLabelWithImplicitTokenLabel() throws Exception {
		String action = "$ID.text;";
		String expecting = "(x!=null?x.getText():null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : x=ID {"+action+"} ;" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testReuseExistingListLabelWithImplicitTokenLabel() throws Exception {
		String action = "$ID.text;";
		String expecting = "(x!=null?x.getText():null);";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : x+=ID {"+action+"} ;" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRuleLabelWithoutOutputOption() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar T;\n"+
			"s : x+=a ;" +
			"a : 'a';\n"+
			"b : 'b';\n"+
			"WS : ' '|'\n';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_LIST_LABEL_INVALID_UNLESS_RETVAL_STRUCT;
		Object expectedArg = "x";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testRuleLabelOnTwoDifferentRulesAST() throws Exception {
		String grammar =
			"grammar T;\n"+
			"options {output=AST;}\n"+
			"s : x+=a x+=b {System.out.println($x);} ;" +
			"a : 'a';\n"+
			"b : 'b';\n"+
			"WS : (' '|'\n') {skip();};\n";
		String expecting = "[a, b]\na b\n";
		String found = execParser("T.g", grammar, "TParser", "TLexer",
								  "s", "a b", false);
		assertEquals(expecting, found);
	}

	@Test public void testRuleLabelOnTwoDifferentRulesTemplate() throws Exception {
		String grammar =
			"grammar T;\n"+
			"options {output=template;}\n"+
			"s : x+=a x+=b {System.out.println($x);} ;" +
			"a : 'a' -> {%{\"hi\"}} ;\n"+
			"b : 'b' -> {%{\"mom\"}} ;\n"+
			"WS : (' '|'\n') {skip();};\n";
		String expecting = "[hi, mom]\n";
		String found = execParser("T.g", grammar, "TParser", "TLexer",
								  "s", "a b", false);
		assertEquals(expecting, found);
	}

	@Test public void testMissingArgs() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : r ;" +
			"r[int i] : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_MISSING_RULE_ARGS;
		Object expectedArg = "r";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testArgsWhenNoneDefined() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : r[32,34] ;" +
			"r : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_RULE_HAS_NO_ARGS;
		Object expectedArg = "r";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testReturnInitValue() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : r ;\n" +
			"r returns [int x=0] : 'a' {$x = 4;} ;\n");
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());

		Rule r = g.getRule("r");
		AttributeScope retScope = r.returnScope;
		List parameters = retScope.getAttributes();
		assertNotNull("missing return action", parameters);
		assertEquals(1, parameters.size());
		String found = parameters.get(0).toString();
		String expecting = "int x=0";
		assertEquals(expecting, found);
	}

	@Test public void testMultipleReturnInitValue() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : r ;\n" +
			"r returns [int x=0, int y, String s=new String(\"foo\")] : 'a' {$x = 4;} ;\n");
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());

		Rule r = g.getRule("r");
		AttributeScope retScope = r.returnScope;
		List parameters = retScope.getAttributes();
		assertNotNull("missing return action", parameters);
		assertEquals(3, parameters.size());
		assertEquals("int x=0", parameters.get(0).toString());
		assertEquals("int y", parameters.get(1).toString());
		assertEquals("String s=new String(\"foo\")", parameters.get(2).toString());
	}

	@Test public void testCStyleReturnInitValue() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : r ;\n" +
			"r returns [int (*x)()=NULL] : 'a' ;\n");
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());

		Rule r = g.getRule("r");
		AttributeScope retScope = r.returnScope;
		List parameters = retScope.getAttributes();
		assertNotNull("missing return action", parameters);
		assertEquals(1, parameters.size());
		String found = parameters.get(0).toString();
		String expecting = "int (*)() x=NULL";
		assertEquals(expecting, found);
	}

	@Test public void testArgsWithInitValues() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : r[32,34] ;" +
			"r[int x, int y=3] : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_ARG_INIT_VALUES_ILLEGAL;
		Object expectedArg = "y";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testArgsOnToken() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ID[32,34] ;" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_ARGS_ON_TOKEN_REF;
		Object expectedArg = "ID";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testArgsOnTokenInLexer() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : 'z' ID[32,34] ;" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_RULE_HAS_NO_ARGS;
		Object expectedArg = "ID";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testLabelOnRuleRefInLexer() throws Exception {
		String action = "$i.text";
		String expecting = "(i!=null?i.getText():null)";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : 'z' i=ID {"+action+"};" +
			"fragment ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "R",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRefToRuleRefInLexer() throws Exception {
		String action = "$ID.text";
		String expecting = "(ID1!=null?ID1.getText():null)";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : 'z' ID {"+action+"};" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "R",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testRefToRuleRefInLexerNoAttribute() throws Exception {
		String action = "$ID";
		String expecting = "ID1";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : 'z' ID {"+action+"};" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "R",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testCharLabelInLexer() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : x='z' ;\n");

		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testCharListLabelInLexer() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : x+='z' ;\n");

		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testWildcardCharLabelInLexer() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : x=. ;\n");

		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testWildcardCharListLabelInLexer() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : x+=. ;\n");

		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testMissingArgsInLexer() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"A : R ;" +
			"R[int i] : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_MISSING_RULE_ARGS;
		Object expectedArg = "R";
		Object expectedArg2 = null;
		// getting a second error @1:12, probably from nextToken
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testLexerRulePropertyRefs() throws Exception {
		String action = "$text $type $line $pos $channel $index $start $stop";
		String expecting = "getText() _type state.tokenStartLine state.tokenStartCharPositionInLine _channel -1 state.tokenStartCharIndex (getCharIndex()-1)";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : 'r' {"+action+"};\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "R",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testLexerLabelRefs() throws Exception {
		String action = "$a $b.text $c $d.text";
		String expecting = "a (b!=null?b.getText():null) c (d!=null?d.getText():null)";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : a='c' b='hi' c=. d=DUH {"+action+"};\n" +
			"DUH : 'd' ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "R",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testSettingLexerRulePropertyRefs() throws Exception {
		String action = "$text $type=1 $line=1 $pos=1 $channel=1 $index";
		String expecting = "getText() _type=1 state.tokenStartLine=1 state.tokenStartCharPositionInLine=1 _channel=1 -1";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"lexer grammar t;\n"+
			"R : 'r' {"+action+"};\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "R",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();

		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testArgsOnTokenInLexerRuleOfCombined() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : R;\n" +
			"R : 'z' ID[32] ;\n" +
			"ID : 'a';\n");

		String lexerGrammarStr = g.getLexerGrammar();
		StringReader sr = new StringReader(lexerGrammarStr);
		Grammar lexerGrammar = new Grammar();
		lexerGrammar.setFileName("<internally-generated-lexer>");
		lexerGrammar.importTokenVocabulary(g);
		lexerGrammar.parseAndBuildAST(sr);
		lexerGrammar.defineGrammarSymbols();
		lexerGrammar.checkNameSpaceAndActions();
		sr.close();

		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, lexerGrammar, "Java");
		lexerGrammar.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_RULE_HAS_NO_ARGS;
		Object expectedArg = "ID";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, lexerGrammar, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testMissingArgsOnTokenInLexerRuleOfCombined() throws Exception {
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : R;\n" +
			"R : 'z' ID ;\n" +
			"ID[int i] : 'a';\n");

		String lexerGrammarStr = g.getLexerGrammar();
		StringReader sr = new StringReader(lexerGrammarStr);
		Grammar lexerGrammar = new Grammar();
		lexerGrammar.setFileName("<internally-generated-lexer>");
		lexerGrammar.importTokenVocabulary(g);
		lexerGrammar.parseAndBuildAST(sr);
		lexerGrammar.defineGrammarSymbols();
		lexerGrammar.checkNameSpaceAndActions();
		sr.close();

		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, lexerGrammar, "Java");
		lexerGrammar.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_MISSING_RULE_ARGS;
		Object expectedArg = "ID";
		Object expectedArg2 = null;
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, lexerGrammar, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	// T R E E S

	@Test public void testTokenLabelTreeProperty() throws Exception {
		String action = "$id.tree;";
		String expecting = "id_tree;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : id=ID {"+action+"} ;\n" +
			"ID : 'a';\n");

		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		ActionTranslator translator =
			new ActionTranslator(generator,
									  "a",
									  new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testTokenRefTreeProperty() throws Exception {
		String action = "$ID.tree;";
		String expecting = "ID1_tree;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ID {"+action+"} ;" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		ActionTranslator translator = new ActionTranslator(generator,"a",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);
	}

	@Test public void testAmbiguousTokenRef() throws Exception {
		String action = "$ID;";
		String expecting = "";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ID ID {"+action+"};" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_NONUNIQUE_REF;
		Object expectedArg = "ID";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testAmbiguousTokenRefWithProp() throws Exception {
		String action = "$ID.text;";
		String expecting = "";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n"+
			"a : ID ID {"+action+"};" +
			"ID : 'a';\n");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();

		int expectedMsgID = ErrorManager.MSG_NONUNIQUE_REF;
		Object expectedArg = "ID";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg);
		checkError(equeue, expectedMessage);
	}

	@Test public void testRuleRefWithDynamicScope() throws Exception {
		String action = "$field::x = $field.st;";
		String expecting = "((field_scope)field_stack.peek()).x = retval.st;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"field\n" +
			"scope { StringTemplate x; }\n" +
			"    :   'y' {"+action+"}\n" +
			"    ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,
																	 "field",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testAssignToOwnRulenameAttr() throws Exception {
		String action = "$rule.tree = null;";
		String expecting = "retval.tree = null;";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"rule\n" +
			"    : 'y' {" + action +"}\n" +
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
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testAssignToOwnParamAttr() throws Exception {
		String action = "$rule.i = 42; $i = 23;";
		String expecting = "i = 42; i = 23;";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"rule[int i]\n" +
			"    : 'y' {" + action +"}\n" +
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
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testIllegalAssignToOwnRulenameAttr() throws Exception {
		String action = "$rule.stop = 0;";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"rule\n" +
			"    : 'y' {" + action +"}\n" +
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
		Object expectedArg = "rule";
		Object expectedArg2 = "stop";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testIllegalAssignToLocalAttr() throws Exception {
		String action = "$tree = null; $st = null; $start = 0; $stop = 0; $text = 0;";
		String expecting = "retval.tree = null; retval.st = null;   ";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"rule\n" +
			"    : 'y' {" + action +"}\n" +
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
		ArrayList expectedErrors = new ArrayList(3);
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, "start", "");
		expectedErrors.add(expectedMessage);
		GrammarSemanticsMessage expectedMessage2 =
			new GrammarSemanticsMessage(expectedMsgID, g, null, "stop", "");
		expectedErrors.add(expectedMessage2);
				GrammarSemanticsMessage expectedMessage3 =
			new GrammarSemanticsMessage(expectedMsgID, g, null, "text", "");
		expectedErrors.add(expectedMessage3);
		checkErrors(equeue, expectedErrors);

		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);
	}

	@Test public void testIllegalAssignRuleRefAttr() throws Exception {
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
		checkError(equeue, expectedMessage);
	}

	@Test public void testIllegalAssignTokenRefAttr() throws Exception {
		String action = "$ID.text = \"test\";";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"ID\n" +
			"    : 'y' ;" +
			"rule\n" +
			"    : ID {" + action +"}\n" +
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
		Object expectedArg = "ID";
		Object expectedArg2 = "text";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		checkError(equeue, expectedMessage);
	}

	@Test public void testAssignToTreeNodeAttribute() throws Exception {
		String action = "$tree.scope = localScope;";
		String expecting = "(()retval.tree).scope = localScope;";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar a;\n" +
			"options { output=AST; }" +
			"rule\n" +
			"@init {\n" +
			"   Scope localScope=null;\n" +
			"}\n" +
			"@after {\n" +
			"   $tree.scope = localScope;\n" +
			"}\n" +
			"   : 'a' -> ^('a')\n" +
			";");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator,
																	 "rule",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
		assertEquals(expecting, found);
	}

	@Test public void testDoNotTranslateAttributeCompare() throws Exception {
		String action = "$a.line == $b.line";
		String expecting = "(a!=null?a.getLine():0) == (b!=null?b.getLine():0)";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
				"lexer grammar a;\n" +
				"RULE:\n" +
				"     a=ID b=ID {" + action + "}" +
				"    ;\n" +
				"ID : 'id';"
		);
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();
		ActionTranslator translator = new ActionTranslator(generator,
																	 "RULE",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
		assertEquals(expecting, found);
	}

	@Test public void testDoNotTranslateScopeAttributeCompare() throws Exception {
		String action = "if ($rule::foo == \"foo\" || 1) { System.out.println(\"ouch\"); }";
		String expecting = "if (((rule_scope)rule_stack.peek()).foo == \"foo\" || 1) { System.out.println(\"ouch\"); }";
		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
				"grammar a;\n" +
				"rule\n" +
				"scope {\n" +
				"   String foo;" +
				"} :\n" +
				"     twoIDs" +
				"    ;\n" +
				"twoIDs:\n" +
				"    ID ID {" + action + "}\n" +
				"    ;\n" +
				"ID : 'id';"
		);
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer();
		ActionTranslator translator = new ActionTranslator(generator,
																	 "twoIDs",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		// check that we didn't use scopeSetAttributeRef int translation!
		boolean foundScopeSetAttributeRef = false;
		for (int i = 0; i < translator.chunks.size(); i++) {
			Object chunk = translator.chunks.get(i);
			if (chunk instanceof StringTemplate) {
				if (((StringTemplate)chunk).getName().equals("scopeSetAttributeRef")) {
					foundScopeSetAttributeRef = true;
				}
			}
		}
		assertFalse("action translator used scopeSetAttributeRef template in comparison!", foundScopeSetAttributeRef);
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
		assertEquals(expecting, found);
	}

	@Test public void testTreeRuleStopAttributeIsInvalid() throws Exception {
		String action = "$r.x; $r.start; $r.stop";
		String expecting = "(r!=null?r.x:0); (r!=null?((CommonTree)r.start):null); $r.stop";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"tree grammar t;\n" +
			"options {ASTLabelType=CommonTree;}\n"+
			"a returns [int x]\n" +
			"  :\n" +
			"  ;\n"+
			"b : r=a {###"+action+"!!!}\n" +
			"  ;");
		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // codegen phase sets some vars we need
		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		int expectedMsgID = ErrorManager.MSG_UNKNOWN_RULE_ATTRIBUTE;
		Object expectedArg = "a";
		Object expectedArg2 = "stop";
		GrammarSemanticsMessage expectedMessage =
			new GrammarSemanticsMessage(expectedMsgID, g, null, expectedArg, expectedArg2);
		System.out.println("equeue:"+equeue);
		checkError(equeue, expectedMessage);
	}

	@Test public void testRefToTextAttributeForCurrentTreeRule() throws Exception {
		String action = "$text";
		String expecting = "input.getTokenStream().toString(\n" +
						   "              input.getTreeAdaptor().getTokenStartIndex(retval.start),\n" +
						   "              input.getTreeAdaptor().getTokenStopIndex(retval.start))";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"tree grammar t;\n" +
			"options {ASTLabelType=CommonTree;}\n" +
			"a : {###"+action+"!!!}\n" +
			"  ;\n");

		Tool antlr = newTool();
		antlr.setOutputDirectory(null); // write to /dev/null
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // codegen phase sets some vars we need
		StringTemplate codeST = generator.getRecognizerST();
		String code = codeST.toString();
		String found = code.substring(code.indexOf("###")+3,code.indexOf("!!!"));
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	@Test public void testTypeOfGuardedAttributeRefIsCorrect() throws Exception {
		String action = "int x = $b::n;";
		String expecting = "int x = ((b_scope)b_stack.peek()).n;";

		ErrorQueue equeue = new ErrorQueue();
		ErrorManager.setErrorListener(equeue);
		Grammar g = new Grammar(
			"grammar t;\n" +
			"s : b ;\n"+
			"b\n" +
			"scope {\n" +
			"  int n;\n" +
			"} : '(' b ')' {"+action+"}\n" + // refers to current invocation's n
			"  ;\n");
		Tool antlr = newTool();
		CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
		g.setCodeGenerator(generator);
		generator.genRecognizer(); // forces load of templates
		ActionTranslator translator = new ActionTranslator(generator, "b",
																	 new antlr.CommonToken(ANTLRParser.ACTION,action),1);
		String rawTranslation =
			translator.translate();
		StringTemplateGroup templates =
			new StringTemplateGroup(".", AngleBracketTemplateLexer.class);
		StringTemplate actionST = new StringTemplate(templates, rawTranslation);
		String found = actionST.toString();
		assertEquals(expecting, found);

		assertEquals("unexpected errors: "+equeue, 0, equeue.errors.size());
	}

	// S U P P O R T

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
		assertTrue("no error; "+expectedMessage.msgID+" expected", equeue.errors.size() > 0);
		assertNotNull("couldn't find expected error: "+expectedMessage.msgID+" in "+equeue, foundMsg);
		assertTrue("error is not a GrammarSemanticsMessage",
				   foundMsg instanceof GrammarSemanticsMessage);
		assertEquals(expectedMessage.arg, foundMsg.arg);
		assertEquals(expectedMessage.arg2, foundMsg.arg2);
	}

	/** Allow checking for multiple errors in one test */
	protected void checkErrors(ErrorQueue equeue,
							   ArrayList expectedMessages)
			throws Exception
	{
		ArrayList messageExpected = new ArrayList(equeue.errors.size());
		for (int i = 0; i < equeue.errors.size(); i++) {
			Message m = (Message)equeue.errors.get(i);
			boolean foundMsg = false;
			for (int j = 0; j < expectedMessages.size(); j++) {
				Message em = (Message)expectedMessages.get(j);
				if (m.msgID==em.msgID && m.arg.equals(em.arg) && m.arg2.equals(em.arg2)) {
					foundMsg = true;
				}
			}
			if (foundMsg) {
				messageExpected.add(i, Boolean.TRUE);
			} else
				messageExpected.add(i, Boolean.FALSE);
		}
		for (int i = 0; i < equeue.errors.size(); i++) {
			assertTrue("unexpected error:" + equeue.errors.get(i), ((Boolean)messageExpected.get(i)).booleanValue());
		}
	}
}
