package org.antlr.gunit.swingui.runner;

import org.antlr.gunit.*;
import org.antlr.gunit.swingui.model.*;

/**
 *
 * @author scai
 */
public class NotifiedTestExecuter extends gUnitExecutor {

    private TestSuite testSuite ;

    public NotifiedTestExecuter(GrammarInfo grammarInfo, ClassLoader loader, String testsuiteDir, TestSuite suite) {
    	super(grammarInfo, loader, testsuiteDir);
        testSuite = suite;
    }

    @Override
    public void onFail(ITestCase failTest) {
        if(failTest == null) throw new IllegalArgumentException("Null fail test");

        final String ruleName = failTest.getTestedRuleName();
        if(ruleName == null) throw new NullPointerException("Null rule name");

        final Rule rule = testSuite.getRule(ruleName);
        final TestCase failCase = (TestCase) rule.getElementAt(failTest.getTestCaseIndex());
        failCase.setPass(false);
        //System.out.println(String.format("[FAIL] %s (%d) ", failTest.getTestedRuleName(), failTest.getTestCaseIndex()));
    }

    @Override
    public void onPass(ITestCase passTest) {
        if(passTest == null) throw new IllegalArgumentException("Null pass test");

        final String ruleName = passTest.getTestedRuleName();
        if(ruleName == null) throw new NullPointerException("Null rule name");
        
        final Rule rule = testSuite.getRule(ruleName);
        final TestCase passCase = (TestCase) rule.getElementAt(passTest.getTestCaseIndex());
        passCase.setPass(true);
        //System.out.println(String.format("[PASS] %s (%d) ", passTest.getTestedRuleName(), passTest.getTestCaseIndex()));
    }
}
