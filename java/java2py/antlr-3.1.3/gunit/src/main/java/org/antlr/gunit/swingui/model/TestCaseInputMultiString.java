
package org.antlr.gunit.swingui.model;


/**
 *
 * @author scai
 */
public class TestCaseInputMultiString implements ITestCaseInput {

    private String script;

    public TestCaseInputMultiString(String text) {
        this.script = text;
    }

    @Override
    public String toString() {
        return "<<" + script + ">>";
    }

    public void setScript(String script) {
        this.script = script;
    }

    public String getScript() {
        return this.script;
    }

}
