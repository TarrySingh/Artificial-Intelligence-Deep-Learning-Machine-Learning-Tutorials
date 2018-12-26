package org.antlr.gunit.swingui;

import org.antlr.gunit.swingui.parsers.StGUnitLexer;
import org.antlr.gunit.swingui.parsers.StGUnitParser;
import org.antlr.gunit.swingui.parsers.ANTLRv3Parser;
import org.antlr.gunit.swingui.parsers.ANTLRv3Lexer;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import org.antlr.gunit.swingui.model.*;
import org.antlr.runtime.*;
import org.antlr.stringtemplate.*;

public class Translator {

    private static String TEMPLATE_FILE = "gunit.stg";
    private static StringTemplateGroup templates;
    
    static  {
        InputStream in = Translator.class.getResourceAsStream(TEMPLATE_FILE);
        Reader rd = new InputStreamReader(in);
        templates = new StringTemplateGroup(rd);
    }


    /* From program model to text. */
    public static String toScript(TestSuite testSuite) {
        if(testSuite == null) return "";
        StringTemplate gUnitScript = templates.getInstanceOf("gUnitFile");
        gUnitScript.setAttribute("testSuite", testSuite);
        return gUnitScript.toString();

    }

    /* From textual script to program model. */
    public static TestSuite toTestSuite(File file, List ruleList) {
        final TestSuite result = new TestSuite();
        try {
            final Reader reader = new BufferedReader(
                    new FileReader(file));
            final StGUnitLexer lexer = new StGUnitLexer(
                    new ANTLRReaderStream(reader));
            final CommonTokenStream tokens = new CommonTokenStream(lexer);
            final StGUnitParser parser = new StGUnitParser(tokens);
            final TestSuiteAdapter adapter = new TestSuiteAdapter(result);
            parser.adapter = adapter;
            parser.gUnitDef();
            result.setTokens(tokens);
            reader.close();

            // if the tested grammar exists in the save directory, load rules.
            final String sGrammarFile = file.getParentFile().getAbsolutePath() 
                    + File.separator + result.getGrammarName() + ".g";
            final File fileGrammar = new File(sGrammarFile);
            if(fileGrammar.exists() && fileGrammar.isFile()) {
                //System.out.println("Found tested grammar file.");
                List<Rule> completeRuleList = loadRulesFromGrammar(fileGrammar);
                //System.out.println(String.format("%d / %d", result.getRuleCount(), completeRuleList.size()));
                for(Rule rule: completeRuleList) {
                    if(!result.hasRule(rule)) {
                        result.addRule(rule);
                        //System.out.println("Add rule:" + rule);
                    }
                }
            } else {
                //System.out.println("Tested grammar not found." + sGrammarFile);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            return result;
        }
    }


    /* Load rules from an ANTLR grammar file. */
    public static List<Rule> loadRulesFromGrammar(File grammarFile) {
        final List<String> ruleNames = new ArrayList<String>();
        try {
            final Reader reader = new BufferedReader(
                    new FileReader(grammarFile));
            final ANTLRv3Lexer lexer = new ANTLRv3Lexer(
                    new ANTLRReaderStream(reader));
            CommonTokenStream tokens = new CommonTokenStream(lexer);
            final ANTLRv3Parser parser = new ANTLRv3Parser(tokens);
            parser.rules = ruleNames;
            parser.grammarDef();
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        final List<Rule> ruleList = new ArrayList<Rule>();
        for(String str: ruleNames) {
            ruleList.add(new Rule(str));
        }

        return ruleList;
    }


    public static class TestSuiteAdapter {
        private TestSuite model ;
        private Rule currentRule;

        public TestSuiteAdapter(TestSuite testSuite) {
            model = testSuite;
        }

        public void setGrammarName(String name) {
            model.setGrammarName(name);
        }

        public void startRule(String name) {
            currentRule = new Rule(name);
        }

        public void endRule() {
            model.addRule(currentRule);
            currentRule = null;
        }

        public void addTestCase(ITestCaseInput in, ITestCaseOutput out) {
            TestCase testCase = new TestCase(in, out);
            currentRule.addTestCase(testCase);
        }

        private static String trimChars(String text, int numOfChars) {
            return text.substring(numOfChars, text.length() - numOfChars);
        }

        public static ITestCaseInput createFileInput(String fileName) {
            if(fileName == null) throw new IllegalArgumentException("null");
            return new TestCaseInputFile(fileName);
        }

        public static ITestCaseInput createStringInput(String line) {
            if(line == null) throw new IllegalArgumentException("null");
            // trim double quotes
            return new TestCaseInputString(trimChars(line, 1));
        }

        public static ITestCaseInput createMultiInput(String text) {
            if(text == null) throw new IllegalArgumentException("null");
            // trim << and >>
            return new TestCaseInputMultiString(trimChars(text, 2));
        }

        public static ITestCaseOutput createBoolOutput(boolean bool) {
            return new TestCaseOutputResult(bool);
        }

        public static ITestCaseOutput createAstOutput(String ast) {
            if(ast == null) throw new IllegalArgumentException("null");
            return new TestCaseOutputAST(ast);
        }

        public static ITestCaseOutput createStdOutput(String text) {
            if(text == null) throw new IllegalArgumentException("null");
            // trim double quotes
            return new TestCaseOutputStdOut(trimChars(text, 1));
        }

        public static ITestCaseOutput createReturnOutput(String text) {
            if(text == null) throw new IllegalArgumentException("null");
            // trim square brackets
            return new TestCaseOutputReturn(trimChars(text, 1));
        }
    }

}
