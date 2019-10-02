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

/** A class which wraps all testsuites for an individual rule */
import java.util.Map;
import java.util.LinkedHashMap;

public class gUnitTestSuite {
	protected String rule = null;			// paeser rule name for unit testing
	protected String lexicalRule = null;	// lexical rule name
	protected String treeRule = null;		// optional, required for testing tree grammar rule
	protected boolean isLexicalRule = false;
	
	/** A map which stores input/output pairs (individual testsuites). 
	 *  In other words, it maps input data for unit test (gUnitTestInput object)
	 *  to an expected output (Token object).
	 */
	protected Map<gUnitTestInput, AbstractTest> testSuites = new LinkedHashMap<gUnitTestInput, AbstractTest>();
	
	public gUnitTestSuite() {
		;
	}
	
	public gUnitTestSuite(String rule) {
		this.rule = rule;
	}
	
	public gUnitTestSuite(String treeRule, String rule) {
		this.rule = rule;
		this.treeRule = treeRule;
	}
	
	public void setRuleName(String ruleName) { this.rule = ruleName; }
	public void setLexicalRuleName(String lexicalRule) { this.lexicalRule = lexicalRule; this.isLexicalRule = true; }
	public void setTreeRuleName(String treeRuleName) { this.treeRule = treeRuleName; }
	
	public String getRuleName() { return this.rule; }
	public String getLexicalRuleName() { return this.lexicalRule; }
	public String getTreeRuleName() { return this.treeRule; }
	public boolean isLexicalRule() { return this.isLexicalRule; }
	
	public void addTestCase(gUnitTestInput input, AbstractTest expect) {
		if ( input!=null && expect!=null ) {
			expect.setTestedRuleName(this.rule);
			expect.setTestCaseIndex(this.testSuites.size());
			this.testSuites.put(input, expect);
		}
	}
	
}
