/*
 [The "BSD licence"]
 Copyright (c) 2005-2008 Terence Parr
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
package org.antlr.tool;

import org.antlr.stringtemplate.StringTemplate;
import antlr.Token;

/** A problem with the symbols and/or meaning of a grammar such as rule
 *  redefinition.
 */
public class GrammarSemanticsMessage extends Message {
	public Grammar g;
	/** Most of the time, we'll have a token such as an undefined rule ref
	 *  and so this will be set.
	 */
	public Token offendingToken;

	public GrammarSemanticsMessage(int msgID,
						  Grammar g,
						  Token offendingToken)
	{
		this(msgID,g,offendingToken,null,null);
	}

	public GrammarSemanticsMessage(int msgID,
						  Grammar g,
						  Token offendingToken,
						  Object arg)
	{
		this(msgID,g,offendingToken,arg,null);
	}

	public GrammarSemanticsMessage(int msgID,
						  Grammar g,
						  Token offendingToken,
						  Object arg,
						  Object arg2)
	{
		super(msgID,arg,arg2);
		this.g = g;
		this.offendingToken = offendingToken;
	}

	public String toString() {
		line = 0;
		column = 0;
		if ( offendingToken!=null ) {
			line = offendingToken.getLine();
			column = offendingToken.getColumn();
		}
		if ( g!=null ) {
			file = g.getFileName();
		}
		StringTemplate st = getMessageTemplate();
		if ( arg!=null ) {
			st.setAttribute("arg", arg);
		}
		if ( arg2!=null ) {
			st.setAttribute("arg2", arg2);
		}
		return super.toString(st);
	}
}
