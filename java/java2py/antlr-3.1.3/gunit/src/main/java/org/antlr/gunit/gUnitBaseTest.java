/*
 [The "BSD licence"]
 Copyright (c) 2007-2008 Leon, Jen-Yuan Su
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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.PrintStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.antlr.runtime.ANTLRFileStream;
import org.antlr.runtime.ANTLRStringStream;
import org.antlr.runtime.CharStream;
import org.antlr.runtime.CommonTokenStream;
import org.antlr.runtime.Lexer;
import org.antlr.runtime.TokenStream;
import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.tree.CommonTreeNodeStream;
import org.antlr.runtime.tree.TreeNodeStream;
import org.antlr.stringtemplate.StringTemplate;

import junit.framework.TestCase;

/** All gUnit-generated JUnit class should extend this class 
 *  which implements the essential methods for triggering
 *  ANTLR parser/tree walker
 */
public abstract class gUnitBaseTest extends TestCase {
	
	public String packagePath;
	public String lexerPath;
	public String parserPath;
	public String treeParserPath;
	
	protected String stdout;
	protected String stderr;
	
	private PrintStream console = System.out;
	private PrintStream consoleErr = System.err;
	
	// Invoke target lexer.rule
	public String execLexer(String testRuleName, String testInput, boolean isFile) throws Exception {
		CharStream input;
		/** Set up ANTLR input stream based on input source, file or String */
		if ( isFile ) {
			String filePath = testInput;
			File testInputFile = new File(filePath);
			// if input test file is not found under the current dir, also try to look for it under the package dir
			if ( !testInputFile.exists() && packagePath!=null ) {
				testInputFile = new File(packagePath, filePath);
				if ( testInputFile.exists() ) filePath = testInputFile.getCanonicalPath();
			}
			input = new ANTLRFileStream(filePath);
		}
		else {
			input = new ANTLRStringStream(testInput);
		}
		Class lexer = null;
		PrintStream ps = null;		// for redirecting stdout later
		PrintStream ps2 = null;		// for redirecting stderr later
        try {
            /** Use Reflection to create instances of lexer and parser */
        	lexer = Class.forName(lexerPath);
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
            	ps2.println("extra text found, '"+input.substring(currentIndex, input.size()-1)+"'");
            }
			
            this.stdout = null;
			this.stderr = null;
            
			if ( err.toString().length()>0 ) {
				this.stderr = err.toString();
				return this.stderr;
			}
			if ( out.toString().length()>0 ) {
				this.stdout = out.toString();
			}
			if ( err.toString().length()==0 && out.toString().length()==0 ) {
				return null;
			}
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
        	if ( e.getCause()!=null ) this.stderr = e.getCause().toString();
			else this.stderr = e.toString();
        	return this.stderr;
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
        return this.stdout;
	}
	
	// Invoke target parser.rule
	public Object execParser(String testRuleName, String testInput, boolean isFile) throws Exception {
		CharStream input;
		/** Set up ANTLR input stream based on input source, file or String */
		if ( isFile ) {
			String filePath = testInput;
			File testInputFile = new File(filePath);
			// if input test file is not found under the current dir, also try to look for it under the package dir
			if ( !testInputFile.exists() && packagePath!=null ) {
				testInputFile = new File(packagePath, filePath);
				if ( testInputFile.exists() ) filePath = testInputFile.getCanonicalPath();
			}
			input = new ANTLRFileStream(filePath);
		}
		else {
			input = new ANTLRStringStream(testInput);
		}
		Class lexer = null;
		Class parser = null;
		PrintStream ps = null;		// for redirecting stdout later
		PrintStream ps2 = null;		// for redirecting stderr later
		try {
			/** Use Reflection to create instances of lexer and parser */
			lexer = Class.forName(lexerPath);
            Class[] lexArgTypes = new Class[]{CharStream.class};				// assign type to lexer's args
            Constructor lexConstructor = lexer.getConstructor(lexArgTypes);
            Object[] lexArgs = new Object[]{input};								// assign value to lexer's args   
            Object lexObj = lexConstructor.newInstance(lexArgs);				// makes new instance of lexer    
            
            CommonTokenStream tokens = new CommonTokenStream((Lexer) lexObj);
            parser = Class.forName(parserPath);
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
                		Class _return = Class.forName(parserPath+"$"+testRuleName+"_return");
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

			this.stdout = null;
			this.stderr = null;
			
			/** Invalid input */
            if ( tokens.index()!=tokens.size() ) {
            	throw new InvalidInputException();
            }
            
			// retVal could be actual return object from rule, stderr or stdout
			if ( err.toString().length()>0 ) {
				this.stderr = err.toString();
				return this.stderr;
			}
			if ( out.toString().length()>0 ) {
				this.stdout = out.toString();
			}
			if ( astString!=null ) {	// Return toStringTree of AST
				return astString;
			}
			else if ( stString!=null ) {// Return toString of ST
				return stString;
			}
			if ( ruleReturn!=null ) {
				return ruleReturn;
			}
			if ( err.toString().length()==0 && out.toString().length()==0 ) {
				return null;
			}
		} catch (ClassNotFoundException e) {
			e.printStackTrace(); System.exit(1);
		} catch (SecurityException e) {
			e.printStackTrace(); System.exit(1);
		} catch (NoSuchMethodException e) {
			e.printStackTrace(); System.exit(1);
		} catch (IllegalAccessException e) {
			e.printStackTrace(); System.exit(1);
		} catch (InvocationTargetException e) {
			if ( e.getCause()!=null ) this.stderr = e.getCause().toString();
			else this.stderr = e.toString();
        	return this.stderr;
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
		return this.stdout;
	}
	
	// Invoke target parser.rule
	public Object execTreeParser(String testTreeRuleName, String testRuleName, String testInput, boolean isFile) throws Exception {
		CharStream input;
		if ( isFile ) {
			String filePath = testInput;
			File testInputFile = new File(filePath);
			// if input test file is not found under the current dir, also try to look for it under the package dir
			if ( !testInputFile.exists() && packagePath!=null ) {
				testInputFile = new File(packagePath, filePath);
				if ( testInputFile.exists() ) filePath = testInputFile.getCanonicalPath();
			}
			input = new ANTLRFileStream(filePath);
		}
		else {
			input = new ANTLRStringStream(testInput);
		}
		Class lexer = null;
		Class parser = null;
		Class treeParser = null;
		PrintStream ps = null;		// for redirecting stdout later
		PrintStream ps2 = null;		// for redirecting stderr later
		try {
			/** Use Reflection to create instances of lexer and parser */
        	lexer = Class.forName(lexerPath);
            Class[] lexArgTypes = new Class[]{CharStream.class};				// assign type to lexer's args
            Constructor lexConstructor = lexer.getConstructor(lexArgTypes);        
            Object[] lexArgs = new Object[]{input};								// assign value to lexer's args   
            Object lexObj = lexConstructor.newInstance(lexArgs);				// makes new instance of lexer    
            
            CommonTokenStream tokens = new CommonTokenStream((Lexer) lexObj);
            
            parser = Class.forName(parserPath);
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
            
            Class _return = Class.forName(parserPath+"$"+testRuleName+"_return");            	
        	Method returnName = _return.getMethod("getTree");
        	CommonTree tree = (CommonTree) returnName.invoke(ruleReturn);

        	// Walk resulting tree; create tree nodes stream first
        	CommonTreeNodeStream nodes = new CommonTreeNodeStream(tree);
        	// AST nodes have payload that point into token stream
        	nodes.setTokenStream(tokens);
        	// Create a tree walker attached to the nodes stream
        	treeParser = Class.forName(treeParserPath);
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
                		Class _treeReturn = Class.forName(treeParserPath+"$"+testTreeRuleName+"_return");
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

			this.stdout = null;
			this.stderr = null;
			
			/** Invalid input */
            if ( tokens.index()!=tokens.size() ) {
            	throw new InvalidInputException();
            }
			
			// retVal could be actual return object from rule, stderr or stdout
			if ( err.toString().length()>0 ) {
				this.stderr = err.toString();
				return this.stderr;
			}
			if ( out.toString().length()>0 ) {
				this.stdout = out.toString();
			}
			if ( astString!=null ) {	// Return toStringTree of AST
				return astString;
			}
			else if ( stString!=null ) {// Return toString of ST
				return stString;
			}
			if ( treeRuleReturn!=null ) {
				return treeRuleReturn;
			}
			if ( err.toString().length()==0 && out.toString().length()==0 ) {
				return null;
			}
		} catch (ClassNotFoundException e) {
			e.printStackTrace(); System.exit(1);
		} catch (SecurityException e) {
			e.printStackTrace(); System.exit(1);
		} catch (NoSuchMethodException e) {
			e.printStackTrace(); System.exit(1);
		} catch (IllegalAccessException e) {
			e.printStackTrace(); System.exit(1);
		} catch (InvocationTargetException e) {
			if ( e.getCause()!=null ) this.stderr = e.getCause().toString();
			else this.stderr = e.toString();
        	return this.stderr;
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
		return stdout;
	}
	
	// Modify the return value if the expected token type is OK or FAIL
	public Object examineExecResult(int tokenType, Object retVal) {	
		if ( tokenType==gUnitParser.OK ) {	// expected Token: OK
			if ( this.stderr==null ) {
				return "OK";
			}
			else {
				return "FAIL, "+this.stderr;
			}
		}
		else if ( tokenType==gUnitParser.FAIL ) {	// expected Token: FAIL
			if ( this.stderr!=null ) {
				return "FAIL";
			}
			else {
				return "OK";
			}
		}
		else {	// return the same object for the other token types
			return retVal;
		}		
	}
	
}
