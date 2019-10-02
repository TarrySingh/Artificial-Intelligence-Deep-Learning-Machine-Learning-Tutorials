
package org.antlr.gunit.swingui.model;

import javax.swing.JComponent;
import javax.swing.JLabel;

/**
 *
 * @author scai
 */
public class TestCaseInputFile implements ITestCaseInput {

    private String fileName;

    public TestCaseInputFile(String file) {
        this.fileName = file;
    }

    public String getLabel() {
        return "FILE:" + fileName;
    }

    public void setScript(String script) {
        this.fileName = script;
    }

    @Override
    public String toString() {
        return fileName;
    }

    public String getScript() {
        return this.fileName;
    }
}