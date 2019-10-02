package org.antlr.gunit.swingui.model;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.antlr.runtime.CommonTokenStream;

public class TestSuite {

    private List<Rule> rules = new ArrayList<Rule>();
    private String grammarName = "";
    private String grammarDir = "";
    private CommonTokenStream tokens;

    private static final String TEST_SUITE_EXT = ".gunit";
    private static final String GRAMMAR_EXT = ".g";

    public TestSuite() {}

    public TestSuite(File grammarFile, List<Rule> rules) {
        final String grammarFileName = grammarFile.getName();
        grammarName = grammarFileName.substring(
                0, grammarFileName.lastIndexOf("."));

        grammarDir = grammarFile.getParent();
    }

    /* Get the gUnit test suite file name. */
    public String getTestSuiteFileName() {
        return grammarDir + File.separator + grammarName + TEST_SUITE_EXT;
    }

    /* Get the ANTLR grammar file name. */
    public String getGrammarFileName() {
        return grammarDir + File.separator + grammarName + GRAMMAR_EXT;
    }

    public void addRule(Rule currentRule) {
        if(currentRule == null) throw new IllegalArgumentException("Null rule");
        rules.add(currentRule);
    }

    // test rule name
    public boolean hasRule(Rule rule) {
        for(Rule r: rules) {
            if(r.getName().equals(rule.getName())) {
                return true;
            }
        }
        return false;
    }

    public int getRuleCount() {
        return rules.size();
    }
    
    public void setRules(List<Rule> newRules) {
        rules.clear();
        rules.addAll(newRules);
    }

    /* GETTERS AND SETTERS */

    public void setGrammarName(String name) { grammarName = name;}

    public String getGrammarName() { return grammarName; }

    public Rule getRule(int index) { return rules.get(index); }

    public CommonTokenStream getTokens() { return tokens; }
    
    public void setTokens(CommonTokenStream ts) { tokens = ts; }

    public Rule getRule(String name) {
        for(Rule rule: rules) {
            if(rule.getName().equals(name)) {
                return rule;
            }
        }
        return null;
    }
    
    // only for stringtemplate use
    public List getRulesForStringTemplate() {return rules;}
    
}
