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
import java.io.File;
import java.io.IOException;

import org.antlr.runtime.*;

/** The main gUnit interpreter entry point. 
 * 	Read a gUnit script, run unit tests or generate a junit file. 
 */
public class Interp {

	public static void main(String[] args) throws IOException, ClassNotFoundException, RecognitionException {
		/** Pull char from where? */
		CharStream input = null;
		/** If the input source is a testsuite file, where is it? */
		String testsuiteDir = System.getProperty("user.dir");
		
	    /** Generate junit codes */
		if ( args.length>0 && args[0].equals("-o") ) {
			if ( args.length==2 ) {
				input = new ANTLRFileStream(args[1]);
				File f = new File(args[1]);
				testsuiteDir = getTestsuiteDir(f.getCanonicalPath(), f.getName());
			}
			else
				input = new ANTLRInputStream(System.in);
			JUnitCodeGen generater = new JUnitCodeGen(parse(input), testsuiteDir);
			generater.compile();
			return;
		}
		
		
		/** Run gunit tests */
		if ( args.length==1 ) {
			input = new ANTLRFileStream(args[0]);
			File f = new File(args[0]);
			testsuiteDir = getTestsuiteDir(f.getCanonicalPath(), f.getName());
		}
		else
			input = new ANTLRInputStream(System.in);
		
		gUnitExecutor executer = new gUnitExecutor(parse(input), testsuiteDir);
		
		System.out.print(executer.execTest());	// unit test result
		
		//return an error code of the number of failures
		System.exit(executer.failures.size() + executer.invalids.size()); 
	}
	
	public static GrammarInfo parse(CharStream input) throws RecognitionException {
		gUnitLexer lexer = new gUnitLexer(input);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		
		GrammarInfo grammarInfo = new GrammarInfo();
		gUnitParser parser = new gUnitParser(tokens, grammarInfo);
		parser.gUnitDef();	// parse gunit script and save elements to grammarInfo
		return grammarInfo;
	}
	
	public static String getTestsuiteDir(String fullPath, String fileName) {
		return fullPath.substring(0, fullPath.length()-fileName.length());
	}
	
}
