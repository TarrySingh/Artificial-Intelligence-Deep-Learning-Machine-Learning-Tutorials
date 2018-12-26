
package org.antlr.gunit.swingui;

import javax.swing.event.ListDataListener;
import org.antlr.gunit.swingui.model.Rule;
import org.antlr.gunit.swingui.images.ImageFactory;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.DefaultListModel;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JScrollPane;
import javax.swing.ListCellRenderer;
import javax.swing.ListModel;
import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionListener;
import org.antlr.gunit.swingui.model.TestSuite;

public class RuleListController implements IController {

    /* Sub-controls */
    private final JList list = new JList();
    private final JScrollPane scroll = new JScrollPane( list,
            JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,
            JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);

    /* Model */
    private ListModel model = null;
    private TestSuite testSuite = null;

    public RuleListController() {
        this.initComponents();
    }

    public JScrollPane getView() {
        return scroll;
    }

    private void setTestSuite(TestSuite newTestSuite) {
        testSuite = newTestSuite;
        model = new RuleListModel();
        list.setModel(model);
    }
    
    public void initialize(TestSuite ts) {
        setTestSuite(ts);
        if(model.getSize() > 0) list.setSelectedIndex(0);
        list.updateUI();
    }


    /**
     * Initialize view.
     */
    private void initComponents() {

        scroll.setViewportBorder(BorderFactory.createEtchedBorder());
        scroll.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createEmptyBorder(), "Rules"));
        scroll.setOpaque(false);

        list.setOpaque(false);
        list.setSelectionMode(ListSelectionModel.SINGLE_INTERVAL_SELECTION);
        list.setLayoutOrientation(JList.VERTICAL);
        list.setCellRenderer(new RuleListItemRenderer());
    }

    public void setListSelectionListener(ListSelectionListener l) {
        this.list.addListSelectionListener(l);
    }

    public Object getModel() {
        return model;
    }


    /* ITEM RENDERER */

    private class RuleListItemRenderer extends JLabel implements ListCellRenderer{

        public RuleListItemRenderer() {
            this.setPreferredSize(new Dimension(50, 18));
        }

        public Component getListCellRendererComponent(
                JList list, Object value, int index,
                boolean isSelected, boolean hasFocus) {

            if(value instanceof Rule) {
                final Rule item = (Rule) value;
                setText(item.toString());
                setForeground(list.getForeground());

                setIcon(item.getNotEmpty() ? ImageFactory.FAV16 : null);

                if(list.getSelectedValue() == item ) {
                    setBackground(Color.LIGHT_GRAY);
                    setOpaque(true);
                } else {
                    setOpaque(false);
                }

            } else {
                this.setText("Error!");
            }
            return this;
        }
    }

    private class RuleListModel implements ListModel {
        
        public RuleListModel() {
            if(testSuite == null) 
                throw new NullPointerException("Null test suite");
        } 
        
        public int getSize() {
            return testSuite.getRuleCount();
        }

        public Object getElementAt(int index) {
            return testSuite.getRule(index);
        }

        public void addListDataListener(ListDataListener l) {}
        public void removeListDataListener(ListDataListener l) {}
    }
}
