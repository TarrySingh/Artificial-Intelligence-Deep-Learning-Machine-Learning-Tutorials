/*
 [The "BSD licence"]
 Copyright (c) 2005 Terence Parr
 Copyright (c) 2006 Kay Roepke (Objective-C runtime)
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
package org.antlr.codegen;

import org.antlr.stringtemplate.StringTemplate;
import org.antlr.tool.Grammar;
import org.antlr.Tool;
import org.antlr.misc.Utils;

import java.io.IOException;

public class ObjCTarget extends Target {
	protected void genRecognizerHeaderFile(Tool tool,
										   CodeGenerator generator,
										   Grammar grammar,
										   StringTemplate headerFileST,
										   String extName)
	throws IOException
	{
		generator.write(headerFileST, grammar.name + Grammar.grammarTypeToFileNameSuffix[grammar.type] + extName);
	}

	public String getTargetCharLiteralFromANTLRCharLiteral(CodeGenerator generator,
														   String literal)
	{
		if  (literal.startsWith("'\\u") ) {
			literal = "0x" +literal.substring(3, 7);
		} else	{
			int c = literal.charAt(1); // TJP
			if  (c < 32 || c > 127) {
				literal  =  "0x" + Integer.toHexString(c);
			}
		}

		return literal;
	}

	/** Convert from an ANTLR string literal found in a grammar file to
	*  an equivalent string literal in the target language.  For Java, this
	*  is the translation 'a\n"' -> "a\n\"".  Expect single quotes
	*  around the incoming literal.  Just flip the quotes and replace
	*  double quotes with \"
	*/
	public String getTargetStringLiteralFromANTLRStringLiteral(CodeGenerator generator,
															   String literal)
	{
		literal = Utils.replace(literal,"\"","\\\"");
		StringBuffer buf = new StringBuffer(literal);
		buf.setCharAt(0,'"');
		buf.setCharAt(literal.length()-1,'"');
		buf.insert(0,'@');
		return buf.toString();
	}

	/** If we have a label, prefix it with the recognizer's name */
	public String getTokenTypeAsTargetLabel(CodeGenerator generator, int ttype) {
		String name = generator.grammar.getTokenDisplayName(ttype);
		// If name is a literal, return the token type instead
		if ( name.charAt(0)=='\'' ) {
			return String.valueOf(ttype);
		}
		return generator.grammar.name + Grammar.grammarTypeToFileNameSuffix[generator.grammar.type] + "_" + name;
		//return super.getTokenTypeAsTargetLabel(generator, ttype);
		//return this.getTokenTextAndTypeAsTargetLabel(generator, null, ttype);
	}

	/** Target must be able to override the labels used for token types. Sometimes also depends on the token text.*/
	public String getTokenTextAndTypeAsTargetLabel(CodeGenerator generator, String text, int tokenType) {
		String name = generator.grammar.getTokenDisplayName(tokenType);
		// If name is a literal, return the token type instead
		if ( name.charAt(0)=='\'' ) {
			return String.valueOf(tokenType);
		}
		String textEquivalent = text == null ? name : text;
		if (textEquivalent.charAt(0) >= '0' && textEquivalent.charAt(0) <= '9') {
			return textEquivalent;
		} else {
			return generator.grammar.name + Grammar.grammarTypeToFileNameSuffix[generator.grammar.type] + "_" + textEquivalent;
		}
	}

}

