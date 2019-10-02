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

import org.antlr.runtime.Token;

/** OutputTest represents a test for not only standard output string, 
 *  but also AST output which is actually a return value from a parser.
 */
public class OutputTest extends AbstractTest {
	private final Token token;
	
	public OutputTest(Token token) {
		this.token = token;
	}

	@Override
	public String getText() {
		return token.getText();
	}

	@Override
	public int getType() {
		return token.getType();
	}

	@Override
	// return ANTLR error msg if test failed
	public String getResult(gUnitTestResult testResult) {
		// Note: we treat the standard output string as a return value also
		if ( testResult.isSuccess() ) return testResult.getReturned();
		else {
			hasErrorMsg = true;
			return testResult.getError();
		}
	}

	@Override
	public String getExpected() {
		return token.getText();
	}
}
