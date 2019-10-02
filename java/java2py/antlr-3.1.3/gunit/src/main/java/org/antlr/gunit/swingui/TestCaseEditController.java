
package org.antlr.gunit.swingui;

import org.antlr.gunit.swingui.model.*;
import org.antlr.gunit.swingui.images.ImageFactory;
import java.awt.*;
import java.awt.event.*;
import java.util.HashMap;
import javax.swing.*;
import javax.swing.event.*;

/**
 *
 * @author scai
 */
public class TestCaseEditController implements IController {

    private JPanel view = new JPanel();

    private JScrollPane scroll;
    private JPanel paneDetail;
    private AbstractEditorPane paneDetailInput, paneDetailOutput;
    private JToolBar toolbar;
    private JList listCases;
    private ListModel listModel ;

    public ActionListener onTestCaseNumberChange;

    /* EDITORS */
    private InputFileEditor editInputFile;
    private InputStringEditor editInputString;
    private InputMultiEditor editInputMulti;
    private OutputResultEditor editOutputResult;
    private OutputAstEditor editOutputAST;
    private OutputStdEditor editOutputStd;
    private OutputReturnEditor editOutputReturn;
    
    private JComboBox comboInputType, comboOutputType;

    /* TYPE NAME */
    private static final String IN_TYPE_STRING = "Single-line Text";
    private static final String IN_TYPE_MULTI = "Multi-line Text";
    private static final String IN_TYPE_FILE = "Disk File";
    private static final String OUT_TYPE_BOOL = "OK or Fail";
    private static final String OUT_TYPE_AST = "AST";
    private static final String OUT_TYPE_STD = "Standard Output";
    private static final String OUT_TYPE_RET = "Return Value";

    private static final String DEFAULT_IN_SCRIPT = "";
    private static final String DEFAULT_OUT_SCRIPT = "";

    private static final Object[] INPUT_TYPE =  {
        IN_TYPE_STRING, IN_TYPE_MULTI, IN_TYPE_FILE
    };

    private static final Object[] OUTPUT_TYPE = {
        OUT_TYPE_BOOL, OUT_TYPE_AST, OUT_TYPE_STD, OUT_TYPE_RET
    };

    /* SIZE */
    private static final int TEST_CASE_DETAIL_WIDTH = 300;
    private static final int TEST_EDITOR_WIDTH = 280;
    private static final int TEST_CASE_DETAIL_HEIGHT = 250;
    private static final int TEST_EDITOR_HEIGHT = 120;

    /* MODEL */
    private Rule currentRule = null;
    private TestCase currentTestCase = null;

    /* END OF MODEL*/

    private static final HashMap<Class, String> TypeNameTable;
    static {
        TypeNameTable = new HashMap<Class, String> ();
        TypeNameTable.put(TestCaseInputString.class, IN_TYPE_STRING);
        TypeNameTable.put(TestCaseInputMultiString.class, IN_TYPE_MULTI);
        TypeNameTable.put(TestCaseInputFile.class, IN_TYPE_FILE);

        TypeNameTable.put(TestCaseOutputResult.class, OUT_TYPE_BOOL);
        TypeNameTable.put(TestCaseOutputAST.class, OUT_TYPE_AST);
        TypeNameTable.put(TestCaseOutputStdOut.class, OUT_TYPE_STD);
        TypeNameTable.put(TestCaseOutputReturn.class, OUT_TYPE_RET);
    }

    //private WorkSpaceView owner;

    public TestCaseEditController(WorkSpaceView workspace) {
        //this.owner = workspace;
        initComponents();
    }

    public TestCaseEditController() {
        initComponents();
    }

    public void OnLoadRule(Rule rule) {
        if(rule == null) throw new IllegalArgumentException("Null");
        this.currentRule = rule;
        this.currentTestCase = null;
        this.listModel = rule;
        this.listCases.setModel(this.listModel);      
    }

    public void setCurrentTestCase(TestCase testCase) {
        if(testCase == null) throw new IllegalArgumentException("Null");
        this.listCases.setSelectedValue(testCase, true);
        this.currentTestCase = testCase;
    }

    public Rule getCurrentRule() {
        return this.currentRule;
    }
    
    private void initComponents() {

        /* CASE LIST */
        listCases = new JList();
        listCases.addListSelectionListener(new TestCaseListSelectionListener());
        listCases.setCellRenderer(listRenderer);
        listCases.setOpaque(false);
        
        scroll = new JScrollPane(listCases);
        scroll.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createEmptyBorder(), "Test Cases"));
        scroll.setOpaque(false);
        scroll.setViewportBorder(BorderFactory.createEtchedBorder());

        /* CASE DETAIL */

        editInputString = new InputStringEditor();
        editInputMulti = new InputMultiEditor();
        editInputFile = new InputFileEditor();

        editOutputResult = new OutputResultEditor();
        editOutputAST = new OutputAstEditor();
        editOutputStd = new OutputStdEditor();
        editOutputReturn = new OutputReturnEditor();
        
        paneDetail = new JPanel();
        paneDetail.setBorder(BorderFactory.createEmptyBorder());
        paneDetail.setOpaque(false);

        comboInputType = new JComboBox(INPUT_TYPE);
        comboInputType.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent event) {
                OnInputTestCaseTypeChanged(comboInputType.getSelectedItem());
            }
        });
        comboOutputType = new JComboBox(OUTPUT_TYPE);
        comboOutputType.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent event) {
                OnOutputTestCaseTypeChanged(comboOutputType.getSelectedItem());
            }
        });
        paneDetailInput = new InputEditorPane(comboInputType);
        paneDetailOutput = new OutputEditorPane(comboOutputType);

        BoxLayout layout = new BoxLayout(paneDetail, BoxLayout.PAGE_AXIS);
        paneDetail.setLayout(layout);
        
        paneDetail.add(this.paneDetailInput);
        paneDetail.add(this.paneDetailOutput);

        /* TOOLBAR */
        toolbar = new JToolBar("Edit TestCases", JToolBar.VERTICAL);
        toolbar.setFloatable(false);
        toolbar.add(new AddTestCaseAction());
        toolbar.add(new RemoveTestCaseAction());

        /* COMPOSITE */
        view.setLayout(new BorderLayout());
        view.setBorder(BorderFactory.createEmptyBorder());
        view.setOpaque(false);
        view.add(toolbar, BorderLayout.WEST);
        view.add(scroll, BorderLayout.CENTER);
        view.add(paneDetail, BorderLayout.EAST);
    }

    private void updateInputEditor() {
        JComponent editor = null;

        if(currentTestCase != null ) {
            ITestCaseInput input = this.currentTestCase.getInput();
            if(input instanceof TestCaseInputString) {
                this.editInputString.setText(input.getScript());
                editor = this.editInputString;
                comboInputType.setSelectedItem(IN_TYPE_STRING);
            } else if(input instanceof TestCaseInputMultiString) {
                this.editInputMulti.setText(input.getScript());
                editor = this.editInputMulti.getView();
                comboInputType.setSelectedItem(IN_TYPE_MULTI);
            } else if(input instanceof TestCaseInputFile) {
                this.editInputFile.setText(input.getScript());
                editor = this.editInputFile;
                comboInputType.setSelectedItem(IN_TYPE_FILE);
            } else {
                throw new Error("Wrong type");
            }
        }
        
        paneDetailInput.setEditor(editor);
    }

    private void updateOutputEditor() {
        JComponent editor = null;
        
        if(currentTestCase != null) {
            
            ITestCaseOutput output = this.currentTestCase.getOutput();

            if(output instanceof TestCaseOutputAST) {

                this.editOutputAST.setText(output.getScript());
                editor = this.editOutputAST.getView();
                comboOutputType.setSelectedItem(OUT_TYPE_AST);

            } else if(output instanceof TestCaseOutputResult) {

                this.editOutputResult.setValue(output.getScript());
                editor = this.editOutputResult;
                comboOutputType.setSelectedItem(OUT_TYPE_BOOL);

            } else if(output instanceof TestCaseOutputStdOut) {

                this.editOutputStd.setText(output.getScript());
                editor = this.editOutputStd.getView();
                comboOutputType.setSelectedItem(OUT_TYPE_STD);

            } else if(output instanceof TestCaseOutputReturn) {

                this.editOutputReturn.setText(output.getScript());
                editor = this.editOutputReturn.getView();
                comboOutputType.setSelectedItem(OUT_TYPE_RET);

            } else {

                throw new Error("Wrong type");
                
            }

        }
        this.paneDetailOutput.setEditor(editor);
    }

    private void OnInputTestCaseTypeChanged(Object inputTypeStr) {
        if(this.currentTestCase != null) {
            ITestCaseInput input ;
            if(inputTypeStr == IN_TYPE_STRING) {
                input = new TestCaseInputString(DEFAULT_IN_SCRIPT);
            } else if(inputTypeStr == IN_TYPE_MULTI) {
                input = new TestCaseInputMultiString(DEFAULT_IN_SCRIPT);
            } else if(inputTypeStr == IN_TYPE_FILE) {
                input = new TestCaseInputFile(DEFAULT_IN_SCRIPT);
            } else {
                throw new Error("Wrong Type");
            }

            if(input.getClass().equals(this.currentTestCase.getInput().getClass()))
                return ;

            this.currentTestCase.setInput(input);
        }
        this.updateInputEditor();
    }

    private void OnOutputTestCaseTypeChanged(Object outputTypeStr) {
        if(this.currentTestCase != null) {

            ITestCaseOutput output ;
            if(outputTypeStr == OUT_TYPE_AST) {
                output = new TestCaseOutputAST(DEFAULT_OUT_SCRIPT);
            } else if(outputTypeStr == OUT_TYPE_BOOL) {
                output = new TestCaseOutputResult(false);
            } else if(outputTypeStr == OUT_TYPE_STD) {
                output = new TestCaseOutputStdOut(DEFAULT_OUT_SCRIPT);
            } else if(outputTypeStr == OUT_TYPE_RET) {
                output = new TestCaseOutputReturn(DEFAULT_OUT_SCRIPT);
            } else {
                throw new Error("Wrong Type");
            }

            if(output.getClass().equals(this.currentTestCase.getOutput().getClass()))
                return ;

            this.currentTestCase.setOutput(output);
        }
        this.updateOutputEditor();
    }


    private void OnTestCaseSelected(TestCase testCase) {
        //if(testCase == null) throw new RuntimeException("Null TestCase");
        this.currentTestCase = testCase;
        updateInputEditor();
        updateOutputEditor();

    }

    private void OnAddTestCase() {
        if(currentRule == null) return;
        
        final TestCase newCase = new TestCase(
                new TestCaseInputString(""),
                new TestCaseOutputResult(true));
        this.currentRule.addTestCase(newCase);
        setCurrentTestCase(newCase);

        this.listCases.setSelectedValue(newCase, true);
        this.listCases.updateUI();
        this.OnTestCaseSelected(newCase);
        this.onTestCaseNumberChange.actionPerformed(null);
    }

    private void OnRemoveTestCase() {
        if(currentTestCase == null) return;
        currentRule.removeElement(currentTestCase);
        listCases.updateUI();

        final TestCase nextActiveCase = listCases.isSelectionEmpty() ?
            null : (TestCase) listCases.getSelectedValue() ;
        OnTestCaseSelected(nextActiveCase);
        this.onTestCaseNumberChange.actionPerformed(null);
    }

    public Object getModel() {
        return currentRule;
    }

    public Component getView() {
        return view;
    }

    /* EDITOR CONTAINER */

    abstract public class AbstractEditorPane extends JPanel {

        private JComboBox combo;
        private JComponent editor;
        private String title;
        private JLabel placeHolder = new JLabel();

        public AbstractEditorPane(JComboBox comboBox, String title) {
            this.combo = comboBox;
            this.editor = placeHolder;
            this.title = title;
            this.initComponents();
        }

        private void initComponents() {
            placeHolder.setPreferredSize(new Dimension(
                    TEST_CASE_DETAIL_WIDTH, TEST_CASE_DETAIL_HEIGHT));
            this.setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            this.add(combo, BorderLayout.NORTH);
            this.add(editor, BorderLayout.CENTER);
            this.setOpaque(false);
            this.setBorder(BorderFactory.createTitledBorder(title));
            this.setPreferredSize(new Dimension(
                    TEST_CASE_DETAIL_WIDTH, TEST_CASE_DETAIL_HEIGHT));
        }

        public void setEditor(JComponent newEditor) {
            if(newEditor == null) newEditor = placeHolder;
            this.remove(editor);
            this.add(newEditor);
            this.editor = newEditor;
            this.updateUI();
        }
    }

    public class InputEditorPane extends AbstractEditorPane {
        public InputEditorPane(JComboBox comboBox) {
            super(comboBox, "Input");
        }
    }

    public class OutputEditorPane extends AbstractEditorPane {
        public OutputEditorPane(JComboBox comboBox) {
            super(comboBox, "Output");
        }
    }

    /* INPUT EDITORS */

    public class InputStringEditor extends JTextField implements CaretListener {
        public InputStringEditor() {
            super();

            this.setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
            this.addCaretListener(this);
        }

        public void caretUpdate(CaretEvent arg0) {
            currentTestCase.getInput().setScript(getText());
            listCases.updateUI();
        }
    }

    public class InputMultiEditor implements CaretListener {
        private JTextArea textArea = new JTextArea(20, 30);
        private JScrollPane scroll = new JScrollPane(textArea,
                JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,
                JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);

        public InputMultiEditor() {
            super();
            scroll.setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
            textArea.addCaretListener(this);
        }

        public void caretUpdate(CaretEvent arg0) {
            currentTestCase.getInput().setScript(getText());
            listCases.updateUI();
        }

        public String getText() {
            return textArea.getText();
        }

        public void setText(String text) {
            textArea.setText(text);
        }

        public JComponent getView() {
            return scroll;
        }
    }

    public class InputFileEditor extends InputStringEditor {};

    public class OutputResultEditor extends JPanel implements ActionListener {
        
        private JToggleButton tbFail, tbOk;

        public OutputResultEditor() {
            super();

            tbFail = new JToggleButton("Fail");
            tbOk = new JToggleButton("OK");
            ButtonGroup group = new ButtonGroup();
            group.add(tbFail);
            group.add(tbOk);

            this.add(tbFail);
            this.add(tbOk);

            this.tbFail.addActionListener(this);
            this.tbOk.addActionListener(this);

            this.setPreferredSize(
                    new Dimension(TEST_EDITOR_WIDTH, 100));
        }

        public void actionPerformed(ActionEvent e) {
            TestCaseOutputResult output =
                    (TestCaseOutputResult) currentTestCase.getOutput();

            if(e.getSource() == tbFail) {
                output.setScript(false);
            } else {
                output.setScript(true);
            }

            listCases.updateUI();
        }

        public void setValue(String value) {
            if(TestCaseOutputResult.OK.equals(value)) {
                this.tbOk.setSelected(true);
            } else {
                this.tbFail.setSelected(true);
            }
        }
    }
    

    public class OutputAstEditor implements CaretListener {
        private JTextArea textArea = new JTextArea(20, 30);
        private JScrollPane scroll = new JScrollPane(textArea,
                JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,
                JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);

        public OutputAstEditor() {
            super();
            scroll.setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
            textArea.addCaretListener(this);
        }

        public void caretUpdate(CaretEvent arg0) {
            currentTestCase.getOutput().setScript(getText());
            listCases.updateUI();
        }

        public void setText(String text) {
            this.textArea.setText(text);
        }

        public String getText() {
            return this.textArea.getText();
        }

        public JScrollPane getView() {
            return this.scroll;
        }
    }


    public class OutputStdEditor extends OutputAstEditor {}
    public class OutputReturnEditor extends OutputAstEditor {}

    /* EVENT HANDLERS */

    private class TestCaseListSelectionListener implements ListSelectionListener {

        public void valueChanged(ListSelectionEvent e) {
            
            if(e.getValueIsAdjusting()) return;
            final JList list = (JList) e.getSource();
            final TestCase value = (TestCase) list.getSelectedValue();
            if(value != null) OnTestCaseSelected(value);
            
        }

    }

    /* ACTIONS */

    private class AddTestCaseAction extends AbstractAction {
        public AddTestCaseAction() {
            super("Add", ImageFactory.ADD);
            putValue(SHORT_DESCRIPTION, "Add a gUnit test case.");
        }
        public void actionPerformed(ActionEvent e) {
            OnAddTestCase();
        }
    }

    private class RemoveTestCaseAction extends AbstractAction {
        public RemoveTestCaseAction() {
            super("Remove", ImageFactory.DELETE);
            putValue(SHORT_DESCRIPTION, "Remove a gUnit test case.");
        }
        public void actionPerformed(ActionEvent e) {
            OnRemoveTestCase();
        }
    }

    /* CELL RENDERERS */

    private static TestCaseListRenderer listRenderer
            = new TestCaseListRenderer();

    private static class TestCaseListRenderer implements ListCellRenderer {

        private static Font IN_FONT = new Font("mono", Font.PLAIN, 12);
        private static Font OUT_FONT = new Font("default", Font.BOLD, 12);

        public static String clamp(String text, int len) {
            if(text.length() > len) {
                return text.substring(0, len - 3).concat("...");
            } else {
                return text;
            }
        }

        public static String clampAtNewLine(String text) {
            int pos = text.indexOf('\n');
            if(pos >= 0) {
                return text.substring(0, pos).concat("...");
            } else {
                return text;
            }
        }

        public Component getListCellRendererComponent(
                JList list, Object value, int index,
                boolean isSelected, boolean hasFocus) {

            final JPanel pane = new JPanel();
            
            if (value instanceof TestCase) {
                final TestCase item = (TestCase) value;

                // create components
                final JLabel labIn = new JLabel(
                        clamp(clampAtNewLine(item.getInput().getScript()), 18));
                final JLabel labOut = new JLabel(
                        clamp(clampAtNewLine(item.getOutput().getScript()), 18));
                labOut.setFont(OUT_FONT);
                labIn.setFont(IN_FONT);

                labIn.setIcon(item.getInput() instanceof TestCaseInputFile ?
                    ImageFactory.FILE16 : ImageFactory.EDIT16);

                pane.setBorder(BorderFactory.createEtchedBorder());
                pane.setLayout(new BoxLayout(pane, BoxLayout.Y_AXIS));
                pane.add(labIn);
                pane.add(labOut);
                pane.setBackground(isSelected ? Color.LIGHT_GRAY : Color.WHITE);
            } 

            return pane;
        }
    }

}
