package org.antlr.gunit.swingui;

import java.awt.Dimension;
import java.awt.FlowLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;

public class StatusBarController implements IController {

    private final JPanel panel = new JPanel();

    private final JLabel labelText = new JLabel("Ready");
    private final JLabel labelRuleName = new JLabel("");
    private final JProgressBar progress = new JProgressBar();
    
    public StatusBarController() {
        initComponents();
    }

    private void initComponents() {
        labelText.setPreferredSize(new Dimension(300, 20));
        labelText.setHorizontalTextPosition(JLabel.LEFT);
        progress.setPreferredSize(new Dimension(100, 15));

        final JLabel labRuleHint = new JLabel("Rule: ");

        FlowLayout layout = new FlowLayout();
        layout.setAlignment(FlowLayout.LEFT);
        panel.setLayout(layout);
        panel.add(labelText);
        panel.add(progress);
        panel.add(labRuleHint);
        panel.add(labelRuleName);
        panel.setOpaque(false);
        panel.setBorder(javax.swing.BorderFactory.createEmptyBorder());

    }

    public void setText(String text) {
        labelText.setText(text);
    }

    public void setRule(String name) {
        this.labelRuleName.setText(name);
    }

    public Object getModel() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public JPanel getView() {
        return panel;
    }

    public void setProgressIndetermined(boolean value) {
        this.progress.setIndeterminate(value);
    }
    
    public void setProgress(int value) {
        this.progress.setIndeterminate(false);
        this.progress.setValue(value);
    }


}
