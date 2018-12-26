/*
 [The "BSD licence"]
 Copyright (c) 2007-2008 Leon Jen-Yuan Su
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
package org.antlr.gunit;

import java.io.*;
import java.util.List;
import java.util.ArrayList;
import java.lang.reflect.*;
import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;
import org.antlr.stringtemplate.CommonGroupLoader;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.StringTemplateGroup;
import org.antlr.stringtemplate.StringTemplateGroupLoader;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;

public class gUnitExecutor implements ITestSuite {
	public GrammarInfo grammarInfo;
	
	private final ClassLoader grammarClassLoader;
	
	private final String testsuiteDir;
	
	public int numOfTest;

	public int numOfSuccess;

	public int numOfFailure;

	private String title;

	public int numOfInvalidInput;

	private String parserName;

	private String lexerName;
	
	public List<AbstractTest> failures;
	public List<AbstractTest> invalids;
	
	private PrintStream console = System.out;
    private PrintStream consoleErr = System.err;
    
    public gUnitExecutor(GrammarInfo grammarInfo, String testsuiteDir) {
    	this( grammarInfo, determineClassLoader(), testsuiteDir);
    }
    
    private static ClassLoader determineClassLoader() {
    	ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
    	if ( classLoader == null ) {
    		classLoader = gUnitExecutor.class.getClassLoader();
    	}
    	return classLoader;
    }
    
	public gUnitExecutor(GrammarInfo grammarInfo, ClassLoader grammarClassLoader, String testsuiteDir) {
		this.grammarInfo = grammarInfo;
		this.grammarClassLoader = grammarClassLoader;
		this.testsuiteDir = testsuiteDir;
		numOfTest = 0;
		numOfSuccess = 0;
		numOfFailure = 0;
		numOfInvalidInput = 0;
		failures = new ArrayList<AbstractTest>();
		invalids = new ArrayList<AbstractTest>();
	}
	
	protected ClassLoader getGrammarClassLoader() {
		return grammarClassLoader;
	}
	
	protected final Class classForName(String name) throws ClassNotFoundException {
		return getGrammarClassLoader().loadClass( name );
	}
	
	public String execTest() throws IOException{
		// Set up string template for testing result
		StringTemplate testResultST = getTemplateGroup().getInstanceOf("testResult");
		try {
			/** Set up appropriate path for parser/lexer if using package */
			if (grammarInfo.getHeader()!=null ) {
				parserName = grammarInfo.getHeader()+"."+grammarInfo.getGrammarName()+"Parser";
				lexerName = grammarInfo.getHeader()+"."+grammarInfo.getGrammarName()+"Lexer";
			}
			else {
				parserName = grammarInfo.getGrammarName()+"Parser";
				lexerName = grammarInfo.getGrammarName()+"Lexer";
			}
			
			/*** Start Unit/Functional Testing ***/
			// Execute unit test of for parser, lexer and tree grammar
			if ( grammarInfo.getTreeGrammarName()!=null ) {
				title = "executing testsuite for tree grammar:"+grammarInfo.getTreeGrammarName()+" walks "+parserName;
			}
			else {
				title = "executing testsuite for grammar:"+grammarInfo.getGrammarName();
			}
			executeTests();
			// End of exection of unit testing
			
			// Fill in the template holes with the test results
			testResultST.setAttribute("title", title);
			testResultST.setAttribute("num_of_test", numOfTest);
			testResultST.setAttribute("num_of_failure", numOfFailure);
			if ( numOfFailure>0 ) {
				testResultST.setAttribute("failure", failures);
			}
			if ( numOfInvalidInput>0 ) {
				testResultST.setAttribute("has_invalid", true);
				testResultST.setAttribute("num_of_invalid", numOfInvalidInput);
				testResultST.setAttribute("invalid", invalids);
			}
		}
		catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
		return testResultST.toString();
	}
	
	private StringTemplateGroup getTemplateGroup() {
		StringTemplateGroupLoader loader = new CommonGroupLoader("org/antlr/gunit", null);
		StringTemplateGroup.registerGroupLoader(loader);
		StringTemplateGroup.registerDefaultLexer(AngleBracketTemplateLexer.class);
		StringTemplateGroup group = StringTemplateGroup.loadGroup("gUnitTestResult");
		return group;
	}
	
	// TODO: throw more specific exceptions
	private gUnitTestResult runCorrectParser(String parserName, String lexerName, String rule, String lexicalRule, String treeRule, gUnitTestInput input) throws Exception
	{
		if ( lexicalRule!=null ) return runLexer(lexerName, lexicalRule, input);
		else if ( treeRule!=null ) return runTreeParser(parserName, lexerName, rule, treeRule, input);
		else return runParser(parserName, lexerName, rule, input);
	}

	private void executeTests() throws Exception {
		for ( gUnitTestSuite ts: grammarInfo.getRuleTestSuites() ) {
			String rule = ts.getRuleName();
			String lexicalRule = ts.getLexicalRuleName();
			String treeRule = ts.getTreeRuleName();
			for ( gUnitTestInput input: ts.testSuites.keySet() ) {	// each rule may contain multiple tests
				numOfTest++;
				// Run parser, and get the return value or stdout or stderr if there is
				gUnitTestResult result = null;
				AbstractTest test = ts.testSuites.get(input);
				try {
					// TODO: create a -debug option to turn on logging, which shows progress of running tests
					//System.out.print(numOfTest + ". Running rule: " + rule + "; input: '" + input.testInput + "'");
					result = runCorrectParser(parserName, lexerName, rule, lexicalRule, treeRule, input);
					// TODO: create a -debug option to turn on logging, which shows progress of running tests
					//System.out.println("; Expecting " + test.getExpected() + "; Success?: " + test.getExpected().equals(test.getResult(result)));
				} catch ( InvalidInputException e) {
					numOfInvalidInput++;
					test.setHeader(rule, lexicalRule, treeRule, numOfTest, input.getLine());
					test.setActual(input.testInput);
					invalids.add(test);
					continue;
				}	// TODO: ensure there's no other exceptions required to be handled here...
				
				String expected = test.getExpected();
				String actual = test.getResult(result);
				test.setActual(actual);
				
				if (actual == null) {
					numOfFailure++;
					test.setHeader(rule, lexicalRule, treeRule, numOfTest, input.getLine());
					test.setActual("null");
					failures.add(test);
					onFail(test);
				}
				// the 2nd condition is used for the assertFAIL test of lexer rule because BooleanTest return err msg instead of 'FAIL' if isLexerTest
				else if ( expected.equals(actual) || (expected.equals("FAIL")&&!actual.equals("OK") ) ) {
					numOfSuccess++;
					onPass(test);
				}
				// TODO: something with ACTIONS - at least create action test type and throw exception.
				else if ( ts.testSuites.get(input).getType()==gUnitParser.ACTION ) {	// expected Token: ACTION
					numOfFailure++;
					test.setHeader(rule, lexicalRule, treeRule, numOfTest, input.getLine());
					test.setActual("\t"+"{ACTION} is not supported in the grammarInfo yet...");
					failures.add(test);
					onFail(test);
				}
				else {
					numOfFailure++;
					test.setHeader(rule, lexicalRule, treeRule, numOfTest, input.getLine());
					failures.add(test);
					onFail(test);
				}
			}	// end of 2nd for-loop: tests for individual rule
		}	// end of 1st for-loop: testsuites for grammar
	}

	// TODO: throw proper exceptions
	protected gUnitTestResult runLexer(String lexerName, String testRuleName, gUnitTestInput testInput) throws Exception {
		CharStream input;
		Class lexer = null;
		PrintStream ps = null;		// for redirecting stdout later
		PrintStream ps2 = null;		// for redirecting stderr later
		try {
			/** Set up ANTLR input stream based on input source, file or String */
			input = getANTLRInputStream(testInput);
		
            /** Use Reflection to create instances of lexer and parser */
        	lexer = classForName(lexerName);
            Class[] lexArgTypes = new Class[]{CharStream.class};				// assign type to lexer's args
            Constructor lexConstructor = lexer.getConstructor(lexArgTypes);        
            Object[] lexArgs = new Object[]{input};								// assign value to lexer's args   
            Object lexObj = lexConstructor.newInstance(lexArgs);				// makes new instance of lexer    
            
            Method ruleName = lexer.getMethod("m"+testRuleName, new Class[0]);
            
            /** Start of I/O Redirecting */
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ByteArrayOutputStream err = new ByteArrayOutputStream();
            ps = new PrintStream(out);
            ps2 = new PrintStream(err);
            System.setOut(ps);
            System.setErr(ps2);
            /** End of redirecting */

            /** Invoke lexer rule, and get the current index in CharStream */
            ruleName.invoke(lexObj, new Object[0]);
            Method ruleName2 = lexer.getMethod("getCharIndex", new Class[0]);
            int currentIndex = (Integer) ruleName2.invoke(lexObj, new Object[0]);
            if ( currentIndex!=input.size() ) {
            	ps2.print("extra text found, '"+input.substring(currentIndex, input.size()-1)+"'");
            }
			
			if ( err.toString().length()>0 ) {
				gUnitTestResult testResult = new gUnitTestResult(false, err.toString(), true);
				testResult.setError(err.toString());
				return testResult;
			}
			String stdout = null;
			if ( out.toString().length()>0 ) {
				stdout = out.toString();
			}
			return new gUnitTestResult(true, stdout, true);
		} catch (IOException e) {
			return getTestExceptionResult(e);
        } catch (ClassNotFoundException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (SecurityException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (NoSuchMethodException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (IllegalArgumentException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (InstantiationException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (IllegalAccessException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (InvocationTargetException e) {	// This exception could be caused from ANTLR Runtime Exception, e.g. MismatchedTokenException
        	return getTestExceptionResult(e);
        } finally {
        	try {
        		if ( ps!=null ) ps.close();
    			if ( ps2!=null ) ps2.close();
    			System.setOut(console);			// Reset standard output
    			System.setErr(consoleErr);		// Reset standard err out
        	} catch (Exception e) {
        		e.printStackTrace();
        	}
        }
        // TODO: verify this:
        throw new Exception("This should be unreachable?");
	}
	
	// TODO: throw proper exceptions
	protected gUnitTestResult runParser(String parserName, String lexerName, String testRuleName, gUnitTestInput testInput) throws Exception {
		CharStream input;
		Class lexer = null;
		Class parser = null;
		PrintStream ps = null;		// for redirecting stdout later
		PrintStream ps2 = null;		// for redirecting stderr later
		try {
			/** Set up ANTLR input stream based on input source, file or String */
			input = getANTLRInputStream(testInput);
			
            /** Use Reflection to create instances of lexer and parser */
        	lexer = classForName(lexerName);
            Class[] lexArgTypes = new Class[]{CharStream.class};				// assign type to lexer's args
            Constructor lexConstructor = lexer.getConstructor(lexArgTypes);        
            Object[] lexArgs = new Object[]{input};								// assign value to lexer's args   
            Object lexObj = lexConstructor.newInstance(lexArgs);				// makes new instance of lexer    
            
            CommonTokenStream tokens = new CommonTokenStream((Lexer) lexObj);
            
            parser = classForName(parserName);
            Class[] parArgTypes = new Class[]{TokenStream.class};				// assign type to parser's args
            Constructor parConstructor = parser.getConstructor(parArgTypes);
            Object[] parArgs = new Object[]{tokens};							// assign value to parser's args  
            Object parObj = parConstructor.newInstance(parArgs);				// makes new instance of parser      
            
            Method ruleName = parser.getMethod(testRuleName);
            
            /** Start of I/O Redirecting */
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ByteArrayOutputStream err = new ByteArrayOutputStream();
            ps = new PrintStream(out);
            ps2 = new PrintStream(err);
            System.setOut(ps);
            System.setErr(ps2);
            /** End of redirecting */

            /** Invoke grammar rule, and store if there is a return value */
            Object ruleReturn = ruleName.invoke(parObj);
            String astString = null;
            String stString = null;
            /** If rule has return value, determine if it contains an AST or a ST */
            if ( ruleReturn!=null ) {
                if ( ruleReturn.getClass().toString().indexOf(testRuleName+"_return")>0 ) {
                	try {	// NullPointerException may happen here...
                		Class _return = classForName(parserName+"$"+testRuleName+"_return");
                		Method[] methods = _return.getDeclaredMethods();
                		for(Method method : methods) {
			                if ( method.getName().equals("getTree") ) {
			                	Method returnName = _return.getMethod("getTree");
		                    	CommonTree tree = (CommonTree) returnName.invoke(ruleReturn);
		                    	astString = tree.toStringTree();
			                }
			                else if ( method.getName().equals("getTemplate") ) {
			                	Method returnName = _return.getMethod("getTemplate");
			                	StringTemplate st = (StringTemplate) returnName.invoke(ruleReturn);
			                	stString = st.toString();
			                }
			            }
                	}
                	catch(Exception e) {
                		System.err.println(e);	// Note: If any exception occurs, the test is viewed as failed.
                	}
                }
            }
            
            /** Invalid input */
            if ( tokens.index()!=tokens.size() ) {
            	//throw new InvalidInputException();
            	ps2.print("Invalid input");
            }
			
			if ( err.toString().length()>0 ) {
				gUnitTestResult testResult = new gUnitTestResult(false, err.toString());
				testResult.setError(err.toString());
				return testResult;
			}
			String stdout = null;
			// TODO: need to deal with the case which has both ST return value and stdout
			if ( out.toString().length()>0 ) {
				stdout = out.toString();
			}
			if ( astString!=null ) {	// Return toStringTree of AST
				return new gUnitTestResult(true, stdout, astString);
			}
			else if ( stString!=null ) {// Return toString of ST
				return new gUnitTestResult(true, stdout, stString);
			}
			
			if ( ruleReturn!=null ) {
				// TODO: currently only works for a single return with int or String value
				return new gUnitTestResult(true, stdout, String.valueOf(ruleReturn));
			}
			return new gUnitTestResult(true, stdout, stdout);
		} catch (IOException e) {
			return getTestExceptionResult(e);
		} catch (ClassNotFoundException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (SecurityException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (NoSuchMethodException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (IllegalArgumentException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (InstantiationException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (IllegalAccessException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (InvocationTargetException e) {	// This exception could be caused from ANTLR Runtime Exception, e.g. MismatchedTokenException
        	return getTestExceptionResult(e);
        } finally {
        	try {
        		if ( ps!=null ) ps.close();
    			if ( ps2!=null ) ps2.close();
    			System.setOut(console);			// Reset standard output
    			System.setErr(consoleErr);		// Reset standard err out
        	} catch (Exception e) {
        		e.printStackTrace();
        	}
        }
        // TODO: verify this:
        throw new Exception("This should be unreachable?");
	}
	
	protected gUnitTestResult runTreeParser(String parserName, String lexerName, String testRuleName, String testTreeRuleName, gUnitTestInput testInput) throws Exception {
		CharStream input;
		String treeParserPath;
		Class lexer = null;
		Class parser = null;
		Class treeParser = null;
		PrintStream ps = null;		// for redirecting stdout later
		PrintStream ps2 = null;		// for redirecting stderr later
		try {
			/** Set up ANTLR input stream based on input source, file or String */
			input = getANTLRInputStream(testInput);
			
			/** Set up appropriate path for tree parser if using package */
			if ( grammarInfo.getHeader()!=null ) {
				treeParserPath = grammarInfo.getHeader()+"."+grammarInfo.getTreeGrammarName();
			}
			else {
				treeParserPath = grammarInfo.getTreeGrammarName();
			}
		
            /** Use Reflection to create instances of lexer and parser */
        	lexer = classForName(lexerName);
            Class[] lexArgTypes = new Class[]{CharStream.class};				// assign type to lexer's args
            Constructor lexConstructor = lexer.getConstructor(lexArgTypes);        
            Object[] lexArgs = new Object[]{input};								// assign value to lexer's args   
            Object lexObj = lexConstructor.newInstance(lexArgs);				// makes new instance of lexer    
            
            CommonTokenStream tokens = new CommonTokenStream((Lexer) lexObj);
            
            parser = classForName(parserName);
            Class[] parArgTypes = new Class[]{TokenStream.class};				// assign type to parser's args
            Constructor parConstructor = parser.getConstructor(parArgTypes);
            Object[] parArgs = new Object[]{tokens};							// assign value to parser's args  
            Object parObj = parConstructor.newInstance(parArgs);				// makes new instance of parser      
            
            Method ruleName = parser.getMethod(testRuleName);

            /** Start of I/O Redirecting */
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ByteArrayOutputStream err = new ByteArrayOutputStream();
            ps = new PrintStream(out);
            ps2 = new PrintStream(err);
            System.setOut(ps);
            System.setErr(ps2);
            /** End of redirecting */

            /** Invoke grammar rule, and get the return value */
            Object ruleReturn = ruleName.invoke(parObj);
            
            Class _return = classForName(parserName+"$"+testRuleName+"_return");            	
        	Method returnName = _return.getMethod("getTree");
        	CommonTree tree = (CommonTree) returnName.invoke(ruleReturn);

        	// Walk resulting tree; create tree nodes stream first
        	CommonTreeNodeStream nodes = new CommonTreeNodeStream(tree);
        	// AST nodes have payload that point into token stream
        	nodes.setTokenStream(tokens);
        	// Create a tree walker attached to the nodes stream
        	treeParser = classForName(treeParserPath);
            Class[] treeParArgTypes = new Class[]{TreeNodeStream.class};		// assign type to tree parser's args
            Constructor treeParConstructor = treeParser.getConstructor(treeParArgTypes);
            Object[] treeParArgs = new Object[]{nodes};							// assign value to tree parser's args  
            Object treeParObj = treeParConstructor.newInstance(treeParArgs);	// makes new instance of tree parser      
        	// Invoke the tree rule, and store the return value if there is
            Method treeRuleName = treeParser.getMethod(testTreeRuleName);
            Object treeRuleReturn = treeRuleName.invoke(treeParObj);

            String astString = null;
            String stString = null;
            /** If tree rule has return value, determine if it contains an AST or a ST */
            if ( treeRuleReturn!=null ) {
                if ( treeRuleReturn.getClass().toString().indexOf(testTreeRuleName+"_return")>0 ) {
                	try {	// NullPointerException may happen here...
                		Class _treeReturn = classForName(treeParserPath+"$"+testTreeRuleName+"_return");
                		Method[] methods = _treeReturn.getDeclaredMethods();
			            for(Method method : methods) {
			                if ( method.getName().equals("getTree") ) {
			                	Method treeReturnName = _treeReturn.getMethod("getTree");
		                    	CommonTree returnTree = (CommonTree) treeReturnName.invoke(treeRuleReturn);
		                        astString = returnTree.toStringTree();
			                }
			                else if ( method.getName().equals("getTemplate") ) {
			                	Method treeReturnName = _return.getMethod("getTemplate");
			                	StringTemplate st = (StringTemplate) treeReturnName.invoke(treeRuleReturn);
			                	stString = st.toString();
			                }
			            }
                	}
                	catch(Exception e) {
                		System.err.println(e);	// Note: If any exception occurs, the test is viewed as failed.
                	}
                }
            }
          
            /** Invalid input */
            if ( tokens.index()!=tokens.size() ) {
            	//throw new InvalidInputException();
            	ps2.print("Invalid input");
            }

			if ( err.toString().length()>0 ) {
				gUnitTestResult testResult = new gUnitTestResult(false, err.toString());
				testResult.setError(err.toString());
				return testResult;
			}
			
			String stdout = null;
			// TODO: need to deal with the case which has both ST return value and stdout
			if ( out.toString().length()>0 ) {
				stdout = out.toString();
			}
			if ( astString!=null ) {	// Return toStringTree of AST
				return new gUnitTestResult(true, stdout, astString);
			}
			else if ( stString!=null ) {// Return toString of ST
				return new gUnitTestResult(true, stdout, stString);
			}
			
			if ( treeRuleReturn!=null ) {
				// TODO: again, currently only works for a single return with int or String value
				return new gUnitTestResult(true, stdout, String.valueOf(treeRuleReturn));
			}
			return new gUnitTestResult(true, stdout, stdout);
		} catch (IOException e) {
			return getTestExceptionResult(e);
		} catch (ClassNotFoundException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (SecurityException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (NoSuchMethodException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (IllegalArgumentException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (InstantiationException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (IllegalAccessException e) {
        	e.printStackTrace(); System.exit(1);
        } catch (InvocationTargetException e) {	// note: This exception could be caused from ANTLR Runtime Exception...
        	return getTestExceptionResult(e);
        } finally {
        	try {
        		if ( ps!=null ) ps.close();
    			if ( ps2!=null ) ps2.close();
    			System.setOut(console);			// Reset standard output
    			System.setErr(consoleErr);		// Reset standard err out
        	} catch (Exception e) {
        		e.printStackTrace();
        	}
        }
        // TODO: verify this:
        throw new Exception("Should not be reachable?");
	}
	
	// Create ANTLR input stream based on input source, file or String
	private CharStream getANTLRInputStream(gUnitTestInput testInput) throws IOException {
		CharStream input;
		if ( testInput.inputIsFile ) {
			String filePath = testInput.testInput;
			File testInputFile = new File(filePath);
			// if input test file is not found under the current dir, try to look for it from dir where the testsuite file locates
			if ( !testInputFile.exists() ) {
				testInputFile = new File(this.testsuiteDir, filePath);
				if ( testInputFile.exists() ) filePath = testInputFile.getCanonicalPath();
				// if still not found, also try to look for it under the package dir
				else if ( grammarInfo.getHeader()!=null ) {
					testInputFile = new File("."+File.separator+grammarInfo.getHeader().replace(".", File.separator), filePath);
					if ( testInputFile.exists() ) filePath = testInputFile.getCanonicalPath();
				}
			}
			input = new ANTLRFileStream(filePath);
		}
		else {
			input = new ANTLRStringStream(testInput.testInput);
		}
		return input;
	}
	
	// set up the cause of exception or the exception name into a gUnitTestResult instance
	private gUnitTestResult getTestExceptionResult(Exception e) {
		gUnitTestResult testResult;
    	if ( e.getCause()!=null ) {
    		testResult = new gUnitTestResult(false, e.getCause().toString(), true);
    		testResult.setError(e.getCause().toString());
    	}
    	else {
    		testResult = new gUnitTestResult(false, e.toString(), true);
    		testResult.setError(e.toString());
    	}
    	return testResult;
	}


    public void onPass(ITestCase passTest) {

    }

    public void onFail(ITestCase failTest) {
        
    }
	
}
