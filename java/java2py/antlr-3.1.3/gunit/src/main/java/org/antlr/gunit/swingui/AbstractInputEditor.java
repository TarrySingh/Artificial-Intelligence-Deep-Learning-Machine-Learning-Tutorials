/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package org.antlr.gunit.swingui;

import org.antlr.gunit.swingui.model.ITestCaseInput;
import java.awt.event.ActionListener;
import javax.swing.JComponent;

/**
 *
 * @author scai
 */
public abstract class AbstractInputEditor {

    protected ITestCaseInput input;
    public void setInput(ITestCaseInput input) {
        this.input = input;
    }

    protected JComponent comp;
    public JComponent getControl() { return comp; }

    abstract public void addActionListener(ActionListener l) ;

}
