/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package org.antlr.gunit.swingui;

import org.antlr.gunit.swingui.images.ImageFactory;
import java.awt.*;
import javax.swing.*;

/**
 *
 * @author scai
 */
public class WorkSpaceView extends JFrame {

    protected JSplitPane splitListClient ;
    protected JTabbedPane tabEditors;
    protected JPanel paneToolBar;
    protected StatusBarController paneStatus;
    protected TestCaseEditController paneEditor;
    protected JToolBar toolbar;
    protected JTextArea txtEditor;
    protected RuleListController listRules;
    protected JMenuBar menuBar;
    protected JScrollPane scrollCode;
    protected JPanel resultPane;

    protected JButton btnOpenGrammar;

    public WorkSpaceView() {
        super();
    }

    protected void initComponents() {

        this.paneEditor = new TestCaseEditController(this);
        this.paneStatus = new StatusBarController();

        this.toolbar = new JToolBar();
        this.toolbar.setBorder(BorderFactory.createEmptyBorder());
        this.toolbar.setFloatable(false);
        this.toolbar.setBorder(BorderFactory.createEmptyBorder());

        this.txtEditor = new JTextArea();
        this.txtEditor.setLineWrap(false);
        this.txtEditor.setFont(new Font("Courier New", Font.PLAIN, 13));
        this.scrollCode = new JScrollPane(txtEditor,
                JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,
                JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        this.scrollCode.setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));

        this.tabEditors = new JTabbedPane();
        this.tabEditors.addTab("Case Editor", ImageFactory.TEXTFILE16, this.paneEditor.getView());
        this.tabEditors.addTab("Script Source", ImageFactory.WINDOW16, this.scrollCode);

        this.listRules = new RuleListController();

        this.splitListClient = new JSplitPane( JSplitPane.HORIZONTAL_SPLIT,
                this.listRules.getView(), this.tabEditors);
        this.splitListClient.setResizeWeight(0.4);
        this.splitListClient.setBorder(BorderFactory.createEmptyBorder());


        
        this.getContentPane().add(this.toolbar, BorderLayout.NORTH);
        this.getContentPane().add(this.splitListClient, BorderLayout.CENTER);
        this.getContentPane().add(this.paneStatus.getView(), BorderLayout.SOUTH);

        // self
        this.setPreferredSize(new Dimension(900, 500));
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

}
