package org.antlr.gunit.swingui.model;

/**
 *
 * @author scai
 */
public class TestCaseOutputStdOut implements ITestCaseOutput {
    private String script;

    public TestCaseOutputStdOut(String text) {
        this.script = text;
    }

    @Override
    public String toString() {
        return String.format(" -> \"%s\"", script);
    }

    public void setScript(String script) {
        this.script = script;
    }

    public String getScript() {
        return this.script;
    }
}
