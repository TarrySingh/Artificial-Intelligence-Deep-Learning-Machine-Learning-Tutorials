/**
 * 
 */
package org.antlr.gunit.swingui.runner;

import java.io.File;
import org.antlr.runtime.*;
import org.antlr.runtime.CharStream;
import org.antlr.gunit.*;
import org.antlr.gunit.swingui.model.TestSuite;

/**
 * Adapter between gUnitEditor Swing GUI and gUnit command-line tool.
 * @author scai
 */
public class gUnitAdapter {

    private ParserLoader loader = new ParserLoader(
            "ini", "/Users/scai/Desktop/gUnitEditor/ini") ;


    public void run(String testSuiteFileName, TestSuite testSuite) {
        if (testSuiteFileName == null || testSuiteFileName.equals(""))
            throw new IllegalArgumentException("Null test suite file name.");
        
        try {

            // Parse gUnit test suite file
            final CharStream input = new ANTLRFileStream(testSuiteFileName);
            final gUnitLexer lexer = new gUnitLexer(input);
            final CommonTokenStream tokens = new CommonTokenStream(lexer);
            final GrammarInfo grammarInfo = new GrammarInfo();
            final gUnitParser parser = new gUnitParser(tokens, grammarInfo);
            parser.gUnitDef();	// parse gunit script and save elements to grammarInfo

            // Get test suite dir
            final File f = new File(testSuiteFileName);
            final String fullPath = f.getCanonicalPath();
            final String filename = f.getName();
            final String testsuiteDir =
                    fullPath.substring(0, fullPath.length()-filename.length());

            // Execute test suite
            final gUnitExecutor executer = new NotifiedTestExecuter(
                    grammarInfo, loader, testsuiteDir, testSuite);
            executer.execTest();
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
