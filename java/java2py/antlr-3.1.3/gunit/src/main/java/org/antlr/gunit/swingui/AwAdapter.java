package org.antlr.gunit.swingui;

import javax.swing.JComponent;
import javax.swing.JSplitPane;
import org.antlr.gunit.swingui.model.TestSuite;

public class AwAdapter {

    private JSplitPane splitMain ;
    private RunnerController runner;
    private TestCaseEditController editor;
    private TestSuite testSuite;

    public AwAdapter() {
        initComponents();
        testSuite = new TestSuite();
    }

    public JComponent getView() {
        return splitMain;
    }

    private void initComponents() {
        runner = new RunnerController();
        editor = new TestCaseEditController();

        splitMain = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT,
                editor.getView(), runner.getView());
    }
}
