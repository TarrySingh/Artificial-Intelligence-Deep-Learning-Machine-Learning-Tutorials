
package org.antlr.gunit.swingui;

import org.antlr.gunit.swingui.runner.gUnitAdapter;
import java.awt.*;
import java.io.IOException;
import org.antlr.gunit.swingui.model.*;
import org.antlr.gunit.swingui.images.ImageFactory;
import java.awt.event.*;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import javax.swing.*;
import javax.swing.event.*;

/**
 *
 * @author scai
 */
public class WorkSpaceController implements IController{

    /* MODEL */
    private TestSuite currentTestSuite;
    private gUnitAdapter adapter = new gUnitAdapter();
    private String testSuiteFileName = null;    // path + file

    /* VIEW */
    private final WorkSpaceView view = new WorkSpaceView();

    /* SUB-CONTROL */
    private final RunnerController runner = new RunnerController();

    public WorkSpaceController() {
        view.resultPane = (JPanel) runner.getView();
        view.initComponents();
        this.initEventHandlers();
        this.initToolbar();
    }

    public void show() {
        this.view.setTitle("gUnitEditor");
        this.view.setVisible(true);
        this.view.pack();
    }

    public Component getEmbeddedView() {
        return view.paneEditor.getView();
    }

    private void initEventHandlers() {
        this.view.tabEditors.addChangeListener(new TabChangeListener());
        this.view.listRules.setListSelectionListener(new RuleListSelectionListener());
        this.view.paneEditor.onTestCaseNumberChange = new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                view.listRules.getView().updateUI();
            }
        };
    }

    private void OnCreateTest() {
        JFileChooser jfc = new JFileChooser();
        jfc.setDialogTitle("Create test suite from grammar");
        jfc.setDialogType(JFileChooser.OPEN_DIALOG);
        if( jfc.showOpenDialog(view) != JFileChooser.APPROVE_OPTION ) return;

        view.paneStatus.setProgressIndetermined(true);
        final File grammarFile = jfc.getSelectedFile();
        //final String sName = grammarFile.getName().substring(
        //        0, grammarFile.getName().length() - 2);
        currentTestSuite = new TestSuite(grammarFile, Translator.loadRulesFromGrammar(grammarFile));
        // trim ".g"
        //currentTestSuite.setGrammarName(sName);
        //currentTestSuite.setRules(Translator.loadRulesFromGrammar(grammarFile));
        view.listRules.initialize(currentTestSuite);
        view.tabEditors.setSelectedIndex(0);
        view.paneStatus.setText("Grammar: " + currentTestSuite.getGrammarName());
        view.paneStatus.setProgressIndetermined(false);

        testSuiteFileName = null;
    }

    private void OnSaveTest() {
        File file;
        view.paneStatus.setProgressIndetermined(true);
        try {

            if(testSuiteFileName == null || testSuiteFileName.equals("")) {
                JFileChooser jfc = new JFileChooser();
                jfc.setDialogTitle("Save test suite");
                jfc.setDialogType(JFileChooser.SAVE_DIALOG);
                if( jfc.showSaveDialog(view) != JFileChooser.APPROVE_OPTION ) return;

                file = jfc.getSelectedFile();
                testSuiteFileName = file.getCanonicalPath();
            } else {
                file = new File(testSuiteFileName);
            }

            FileWriter fw = new FileWriter(file);
            fw.write(Translator.toScript(currentTestSuite));
            fw.flush();
            fw.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        view.paneStatus.setProgressIndetermined(false);

    }

    private void OnSaveAsTest() {

    }

    private void OnOpenTest()  {

        JFileChooser jfc = new JFileChooser();
        jfc.setDialogTitle("Open existing gUnit test suite");
        jfc.setDialogType(JFileChooser.OPEN_DIALOG);
        if( jfc.showOpenDialog(view) != JFileChooser.APPROVE_OPTION ) return;

        final File testSuiteFile = jfc.getSelectedFile();
        try {
            testSuiteFileName = testSuiteFile.getCanonicalPath();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        view.paneStatus.setProgressIndetermined(true);

        currentTestSuite = Translator.toTestSuite(testSuiteFile, new ArrayList());
        view.listRules.initialize(currentTestSuite);
        view.paneStatus.setText(currentTestSuite.getGrammarName());
        view.tabEditors.setSelectedIndex(0);

        view.paneStatus.setProgressIndetermined(false);
    }

    private void OnSelectRule(Rule rule) {
        if(rule == null) throw new IllegalArgumentException("Null");
        this.view.paneEditor.OnLoadRule(rule);
        this.view.paneStatus.setRule(rule.getName());

        // run result
        this.runner.OnShowRuleResult(rule);
    }

    private void OnSelectTextPane() {
        Thread worker = new Thread () {
            @Override
            public void run() {
                view.paneStatus.setProgressIndetermined(true);
                view.txtEditor.setText(
                    Translator.toScript(currentTestSuite));
                view.paneStatus.setProgressIndetermined(false);
            }
        };

        worker.start();
    }

    private void OnRunTest() {
        if(currentTestSuite == null) return;
        adapter.run(testSuiteFileName, currentTestSuite);
        view.tabEditors.addTab("Test Result", ImageFactory.FILE16, runner.getView());
        runner.OnShowSuiteResult(currentTestSuite);
    }

    private void initToolbar() {
        view.toolbar.add(new JButton(new CreateAction()));
        view.toolbar.add(new JButton(new OpenAction()));
        view.toolbar.add(new JButton(new SaveAction()));
        view.toolbar.add(new JButton(new RunAction()));

    }

    public Object getModel() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public Component getView() {
        return view;
    }


    /** Event handler for rule list selection. */
    private class RuleListSelectionListener implements ListSelectionListener {
        public void valueChanged(ListSelectionEvent event) {
            if(event.getValueIsAdjusting()) return;
            final JList list = (JList) event.getSource();
            final Rule rule = (Rule) list.getSelectedValue();
            if(rule != null) OnSelectRule(rule);
        }
    }


    /** Event handler for switching between editor view and script view. */
    public class TabChangeListener implements ChangeListener {
        public void stateChanged(ChangeEvent evt) {
            if(view.tabEditors.getSelectedIndex() == 1) {
                OnSelectTextPane();
            }
        }
        
    }







    

    /** Create test suite action. */
    private class CreateAction extends AbstractAction {
        public CreateAction() {
            super("Create", ImageFactory.ADDFILE);
            putValue(SHORT_DESCRIPTION, "Create a test suite from an ANTLR grammar");
        }
        public void actionPerformed(ActionEvent e) {
            OnCreateTest();
        }
    }


    /** Save test suite action. */
    private class SaveAction extends AbstractAction {
        public SaveAction() {
            super("Save", ImageFactory.SAVE);
            putValue(SHORT_DESCRIPTION, "Save the test suite");
        }
        public void actionPerformed(ActionEvent e) {
            OnSaveTest();
        }
    }

    /** Save as test suite action. */
    private class SaveAsAction extends AbstractAction {
        public SaveAsAction() {
            super("Save as", ImageFactory.SAVEAS);
            putValue(SHORT_DESCRIPTION, "Save a copy");
        }
        public void actionPerformed(ActionEvent e) {
            OnSaveAsTest();
        }
    }

    /** Open test suite action. */
    private class OpenAction extends AbstractAction {
        public OpenAction() {
            super("Open", ImageFactory.OPEN);
            putValue(SHORT_DESCRIPTION, "Open an existing test suite");
            putValue(ACCELERATOR_KEY, KeyStroke.getKeyStroke(
                    KeyEvent.VK_O, InputEvent.CTRL_MASK));
        }
        public void actionPerformed(ActionEvent e) {
            OnOpenTest();
        }
    }

    /** Run test suite action. */
    private class RunAction extends AbstractAction {
        public RunAction() {
            super("Run", ImageFactory.NEXT);
            putValue(SHORT_DESCRIPTION, "Run the current test suite");
            putValue(ACCELERATOR_KEY, KeyStroke.getKeyStroke(
                    KeyEvent.VK_R, InputEvent.CTRL_MASK));
        }
        public void actionPerformed(ActionEvent e) {
            OnRunTest();
        }
    }
}
