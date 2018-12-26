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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GrammarInfo {

	private String grammarName;					// targeted grammar for unit test
	private String treeGrammarName = null;		// optional, required for testing tree grammar
	private String header = null;				// optional, required if using java package
	private List<gUnitTestSuite> ruleTestSuites = new ArrayList<gUnitTestSuite>();	// testsuites for each testing rule
	private StringBuffer unitTestResult = new StringBuffer();
	
	public String getGrammarName() {
		return grammarName;
	}
	
	public void setGrammarName(String grammarName) {
		this.grammarName = grammarName;
	}

	public String getTreeGrammarName() {
		return treeGrammarName;
	}

	public void setTreeGrammarName(String treeGrammarName) {
		this.treeGrammarName = treeGrammarName;
	}

	public String getHeader() {
		return header;
	}

	public void setHeader(String header) {
		this.header = header;
	}

	public List<gUnitTestSuite> getRuleTestSuites() {
		// Make this list unmodifiable so that we can refactor knowing it's not changed.
		return Collections.unmodifiableList(ruleTestSuites);
	}
	
	public void addRuleTestSuite(gUnitTestSuite testSuite) {
		this.ruleTestSuites.add(testSuite);
	}
	
	public void appendUnitTestResult(String result) {
		this.unitTestResult.append(result);
	}

	// We don't want people messing with the string buffer here, so don't return it.
	public String getUnitTestResult() {
		return unitTestResult.toString();
	}

	public void setUnitTestResult(StringBuffer unitTestResult) {
		this.unitTestResult = unitTestResult;
	}
}
