/*
 [The "BSD licence"]
 Copyright (c) 2005-2006 Terence Parr
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
import org.antlr.stringtemplate.StringTemplateGroup;
import org.antlr.tool.Grammar;
import org.antlr.Tool;

import java.io.IOException;

public class CPPTarget extends Target {
	
	public String escapeChar( int c ) {
		// System.out.println("CPPTarget.escapeChar("+c+")");
		switch (c) {
		case '\n' : return "\\n";
		case '\t' : return "\\t";
		case '\r' : return "\\r";
		case '\\' : return "\\\\";
		case '\'' : return "\\'";
		case '"' :  return "\\\"";
		default :
			if ( c < ' ' || c > 126 )
			{
				if (c > 255)
				{
					String s = Integer.toString(c,16);
					// put leading zeroes in front of the thing..
					while( s.length() < 4 )
						s = '0' + s;
					return "\\u" + s;
				}
				else {
					return "\\" + Integer.toString(c,8);
				}
			}
			else {
				return String.valueOf((char)c);
			}
		}
	}

	/** Converts a String into a representation that can be use as a literal
	 * when surrounded by double-quotes.
	 *
	 * Used for escaping semantic predicate strings for exceptions.
	 *
	 * @param s The String to be changed into a literal
	 */
	public String escapeString(String s)
	{
		StringBuffer retval = new StringBuffer();
		for (int i = 0; i < s.length(); i++) {
			retval.append(escapeChar(s.charAt(i)));
		}

		return retval.toString();
	}

	protected void genRecognizerHeaderFile(Tool tool,
										   CodeGenerator generator,
										   Grammar grammar,
										   StringTemplate headerFileST,
										   String extName)
		throws IOException
	{
		StringTemplateGroup templates = generator.getTemplates();
		generator.write(headerFileST, grammar.name+extName);
	}

	/** Convert from an ANTLR char literal found in a grammar file to
	 *  an equivalent char literal in the target language.  For Java, this
	 *  is the identify translation; i.e., '\n' -> '\n'.  Most languages
	 *  will be able to use this 1-to-1 mapping.  Expect single quotes
	 *  around the incoming literal.
	 *  Depending on the charvocabulary the charliteral should be prefixed with a 'L'
	 */
	public String getTargetCharLiteralFromANTLRCharLiteral( CodeGenerator codegen, String literal) {
		int c = Grammar.getCharValueFromGrammarCharLiteral(literal);
		String prefix = "'";
		if( codegen.grammar.getMaxCharValue() > 255 )
			prefix = "L'";
		else if( (c & 0x80) != 0 )	// if in char mode prevent sign extensions
			return ""+c;
		return prefix+escapeChar(c)+"'";
	}

	/** Convert from an ANTLR string literal found in a grammar file to
	 *  an equivalent string literal in the target language.  For Java, this
	 *  is the identify translation; i.e., "\"\n" -> "\"\n".  Most languages
	 *  will be able to use this 1-to-1 mapping.  Expect double quotes 
	 *  around the incoming literal.
	 *  Depending on the charvocabulary the string should be prefixed with a 'L'
	 */
	public String getTargetStringLiteralFromANTLRStringLiteral( CodeGenerator codegen, String literal) {
		StringBuffer buf = Grammar.getUnescapedStringFromGrammarStringLiteral(literal);
		String prefix = "\"";
		if( codegen.grammar.getMaxCharValue() > 255 )
			prefix = "L\"";
		return prefix+escapeString(buf.toString())+"\"";
	}
	/** Character constants get truncated to this value.
	 * TODO: This should be derived from the charVocabulary. Depending on it
	 * being 255 or 0xFFFF the templates should generate normal character
	 * constants or multibyte ones.
	 */
	public int getMaxCharValue( CodeGenerator codegen ) {
		int maxval = 255; // codegen.grammar.get????();
		if ( maxval <= 255 )
			return 255;
		else
			return maxval;
	}
}
