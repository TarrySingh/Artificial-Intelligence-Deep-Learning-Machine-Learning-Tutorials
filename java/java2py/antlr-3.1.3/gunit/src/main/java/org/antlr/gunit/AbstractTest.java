/*
 [The "BSD license"]
 Copyright (c) 2007 Kenny MacDermid
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

public abstract class AbstractTest implements ITestCase {
	// store essential individual test result for string template
	protected String header;
	protected String actual;
	
	protected boolean hasErrorMsg;
	
	private String testedRuleName;
	private int testCaseIndex;
	
	// TODO: remove these. They're only used as part of a refactor to keep the
	//       code cleaner. It is a mock-instanceOf() replacement.
	public abstract int getType();
	public abstract String getText();
	
	public abstract String getExpected();
	// return an escaped string of the expected result
	public String getExpectedResult() {
		String expected = getExpected();
		if ( expected!=null ) expected = JUnitCodeGen.escapeForJava(expected);
		return expected;
	}
	public abstract String getResult(gUnitTestResult testResult);
	public String getHeader() { return this.header; }
	public String getActual() { return this.actual; }
	// return an escaped string of the actual result
	public String getActualResult() {
		String actual = getActual();
		// there is no need to escape the error message from ANTLR 
		if ( actual!=null && !hasErrorMsg ) actual = JUnitCodeGen.escapeForJava(actual);
		return actual;
	}
	
	public String getTestedRuleName() { return this.testedRuleName; }
	public int getTestCaseIndex() { return this.testCaseIndex; }
	
	public void setHeader(String rule, String lexicalRule, String treeRule, int numOfTest, int line) {
		StringBuffer buf = new StringBuffer();
		buf.append("test" + numOfTest + " (");
		if ( treeRule!=null ) {
			buf.append(treeRule+" walks ");
		}
		if ( lexicalRule!=null ) {
			buf.append(lexicalRule + ", line"+line+")" + " - ");
		}
		else buf.append(rule + ", line"+line+")" + " - ");
		this.header = buf.toString();
	}
	public void setActual(String actual) { this.actual = actual; }
	
	public void setTestedRuleName(String testedRuleName) { this.testedRuleName = testedRuleName; }
	public void setTestCaseIndex(int testCaseIndex) { this.testCaseIndex = testCaseIndex; }
	
}
