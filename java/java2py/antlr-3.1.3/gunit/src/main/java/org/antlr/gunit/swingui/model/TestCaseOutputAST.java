
package org.antlr.gunit.swingui.model;

/**
 *
 * @author scai
 */
public class TestCaseOutputAST implements ITestCaseOutput {

    private String treeString;

    public TestCaseOutputAST(String script) {
        this.treeString = script;
    }

    public void setScript(String script) {
        this.treeString = script;
    }

    public String getScript() {
        return this.treeString;
    }


    @Override
    public String toString() {
        return String.format(" -> %s", treeString);
    }

}
