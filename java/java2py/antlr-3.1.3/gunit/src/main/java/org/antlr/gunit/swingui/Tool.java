package org.antlr.gunit.swingui;

import java.io.IOException;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;

public class Tool {

    public static void main(String[] args) throws IOException {
        try {
            UIManager.setLookAndFeel( UIManager.getSystemLookAndFeelClassName());
        }
        catch (Exception e) {
        }


        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                WorkSpaceController control = new WorkSpaceController();
                control.show();
            }
        });
    }



}
