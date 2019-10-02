package org.antlr.gunit.swingui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTree;
import javax.swing.event.TreeModelListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeCellRenderer;
import javax.swing.tree.TreeModel;
import javax.swing.tree.TreePath;
import org.antlr.gunit.swingui.images.ImageFactory;
import org.antlr.gunit.swingui.model.*;

/**
 *
 * @author scai
 */
public class RunnerController implements IController {

    /* MODEL */
    //private TestSuite testSuite;

    /* VIEW */
    private RunnerView view = new RunnerView();
    public class RunnerView extends JPanel {
        
        private JTextArea textArea = new JTextArea();

        private JTree tree = new JTree();

        private JScrollPane scroll = new JScrollPane(tree,
                JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,
                JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);

        public void initComponents() {
            //textArea.setOpaque(false);
            tree.setOpaque(false);
            scroll.setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
            scroll.setOpaque(false);
            this.setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            this.add(scroll);
            this.setBorder(BorderFactory.createEmptyBorder());
            this.setOpaque(false);
        }

    };

    public RunnerController() {
    }

    public Object getModel() {
        return null;
    }

    public Component getView() {
        return view;
    }

    public void update() {
        view.initComponents();
    }

    public void OnShowSuiteResult(TestSuite suite) {
        update();
        view.tree.setModel(new RunnerTreeModel(suite));
        view.tree.setCellRenderer(new RunnerTreeRenderer());
    }

    public void OnShowRuleResult(Rule rule) {
        update();
        
        
        
        /*
        StringBuffer result = new StringBuffer();

        result.append("Testing result for rule: " + rule.getName());
        result.append("\n--------------------\n\n");

        for(TestCase testCase: rule.getTestCases()){
            result.append(testCase.isPass() ? "PASS" : "FAIL");
            result.append("\n");
        }
        result.append("\n--------------------\n");
        view.textArea.setText(result.toString());
                  */
    }



    private class TestSuiteTreeNode extends DefaultMutableTreeNode {

        private TestSuite data ;

        public TestSuiteTreeNode(TestSuite suite) {
            super(suite.getGrammarName());
            for(int i=0; i<suite.getRuleCount(); ++i) {
                final Rule rule = suite.getRule(i);
                if(rule.getNotEmpty()) this.add(new TestGroupTreeNode(rule));
            }
            data = suite;
        }

        @Override
        public String toString() {
            return String.format("%s (%d test groups)",
                    data.getGrammarName(),
                    this.getChildCount());
        }

    } ;

    private class TestGroupTreeNode extends DefaultMutableTreeNode {

        private Rule data;
        private boolean hasFail = false;

        private TestGroupTreeNode(Rule rule) {
            super(rule.getName());
            for(TestCase tc: rule.getTestCases()) {
                this.add(new TestCaseTreeNode(tc));
            }

            data = rule;
         }

        @Override
        public String toString() {
            int iPass = 0;
            int iFail = 0;
            for(TestCase tc: data.getTestCases()) {
                if(tc.isPass())
                    ++iPass;
                else
                    ++iFail;
            }

            hasFail = iFail > 0;

            return String.format("%s (pass %d, fail %d)",
                data.getName(), iPass, iFail);
        }
    } ;

    private class TestCaseTreeNode extends DefaultMutableTreeNode {

        private TestCase data;

        private TestCaseTreeNode(TestCase tc) {
            super(tc.toString());
            data = tc;
        }
    } ;

    private class RunnerTreeModel extends DefaultTreeModel {

        public RunnerTreeModel(TestSuite testSuite) {
            super(new TestSuiteTreeNode(testSuite));
        }
    }

    private class RunnerTreeRenderer implements TreeCellRenderer {

        public Component getTreeCellRendererComponent(JTree tree, Object value,
                boolean selected, boolean expanded, boolean leaf, int row,
                boolean hasFocus) {

            JLabel label = new JLabel();

            if(value instanceof TestSuiteTreeNode) {

                label.setText(value.toString());
                label.setIcon(ImageFactory.TESTSUITE);

            } else if(value instanceof TestGroupTreeNode) {

                TestGroupTreeNode node = (TestGroupTreeNode) value;
                label.setText(value.toString());
                label.setIcon( node.hasFail ? ImageFactory.TESTGROUPX : ImageFactory.TESTGROUP);

            } else if(value instanceof TestCaseTreeNode) {

                TestCaseTreeNode node = (TestCaseTreeNode) value;
                label.setIcon( (node.data.isPass())?
                    ImageFactory.RUN_PASS : ImageFactory.RUN_FAIL);
                label.setText(value.toString());

            } else {
                throw new IllegalArgumentException(
                    "Invalide tree node type + " + value.getClass().getName());
            }

            return label;
            
        }
        
    }

}
