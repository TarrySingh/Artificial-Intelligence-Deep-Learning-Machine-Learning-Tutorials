/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package org.antlr.gunit.swingui.model;

import java.util.ArrayList;
import java.util.List;
import javax.swing.DefaultListModel;

/**
 * ANTLR v3 Rule Information.
 * @author scai
 */
public class Rule extends DefaultListModel {

    private String name;

    public Rule(String name) {
        this.name = name;
    }

    public String getName() { return name; }

    public boolean getNotEmpty() {
        return !this.isEmpty();
    }

    @Override
    public String toString() {
        return this.name;
    }

    public void addTestCase(TestCase newItem) {
        this.addElement(newItem);
    }
    
    // for string template
    public List<TestCase> getTestCases() {
        List<TestCase> result = new ArrayList<TestCase>();
        for(int i=0; i<this.size(); i++) {
            result.add((TestCase)this.getElementAt(i));
        }
        return result;
    }
}
