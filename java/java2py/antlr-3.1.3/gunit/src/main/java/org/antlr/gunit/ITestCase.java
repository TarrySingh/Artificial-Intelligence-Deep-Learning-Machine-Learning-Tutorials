/*
 [The "BSD license"]
 Copyright (c) 2009 Shaoting Cai
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

/**
 * ITestCase object locates one test case in a gUnit script by specifying the
 * tested rule and the index number of the test case in that group.
 *
 * For example:
 * ----------------------
 * ...
 * varDef:
 * "int i;" OK
 * "float 2f;" FAIL
 * ...
 * ----------------------
 * The "testedRuleName" for these two test cases will be "varDef".
 * The "index" for the "int"-test will be 0.
 * The "index" for the "float"-test will be 1.  And so on.
 *
 * @see ITestSuite
 */
public interface ITestCase {

    /**
     * Get the name of the rule that is tested by this test case.
     * @return name of the tested rule.
     */
    public String getTestedRuleName();

    /**
     * Get the index of the test case in the test group for a rule. Starting
     * from 0.
     * @return index number of the test case.
     */
    public int getTestCaseIndex();
	
}
