
package org.antlr.gunit.swingui.model;

/**
 *
 * @author scai
 */
public class TestCaseOutputResult implements ITestCaseOutput {

    public static String OK = "OK";
    public static String FAIL = "FAIL";

    private boolean success ;

    public TestCaseOutputResult(boolean result) {
        this.success = result;
    }

    @Override
    public String toString() {
        return getScript();
    }

    public String getScript() {
        return success ? OK : FAIL;
    }

    public void setScript(boolean value) {
        this.success = value;
    }

    public void setScript(String script) {
        success = Boolean.parseBoolean(script);
    }

}
