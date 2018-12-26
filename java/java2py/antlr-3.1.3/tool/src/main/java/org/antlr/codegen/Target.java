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

import org.antlr.Tool;
import org.antlr.analysis.Label;
import org.antlr.misc.Utils;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.tool.Grammar;

import java.io.IOException;
import java.util.List;

/** The code generator for ANTLR can usually be retargeted just by providing
 *  a new X.stg file for language X, however, sometimes the files that must
 *  be generated vary enough that some X-specific functionality is required.
 *  For example, in C, you must generate header files whereas in Java you do not.
 *  Other languages may want to keep DFA separate from the main
 *  generated recognizer file.
 *
 *  The notion of a Code Generator target abstracts out the creation
 *  of the various files.  As new language targets get added to the ANTLR
 *  system, this target class may have to be altered to handle more
 *  functionality.  Eventually, just about all language generation issues
 *  will be expressible in terms of these methods.
 *
 *  If org.antlr.codegen.XTarget class exists, it is used else
 *  Target base class is used.  I am using a superclass rather than an
 *  interface for this target concept because I can add functionality
 *  later without breaking previously written targets (extra interface
 *  methods would force adding dummy functions to all code generator
 *  target classes).
 *
 */
public class Target {

	/** For pure strings of Java 16-bit unicode char, how can we display
	 *  it in the target language as a literal.  Useful for dumping
	 *  predicates and such that may refer to chars that need to be escaped
	 *  when represented as strings.  Also, templates need to be escaped so
	 *  that the target language can hold them as a string.
	 *
	 *  I have defined (via the constructor) the set of typical escapes,
	 *  but your Target subclass is free to alter the translated chars or
	 *  add more definitions.  This is nonstatic so each target can have
	 *  a different set in memory at same time.
	 */
	protected String[] targetCharValueEscape = new String[255];

	public Target() {
		targetCharValueEscape['\n'] = "\\n";
		targetCharValueEscape['\r'] = "\\r";
		targetCharValueEscape['\t'] = "\\t";
		targetCharValueEscape['\b'] = "\\b";
		targetCharValueEscape['\f'] = "\\f";
		targetCharValueEscape['\\'] = "\\\\";
		targetCharValueEscape['\''] = "\\'";
		targetCharValueEscape['"'] = "\\\"";
	}

	protected void genRecognizerFile(Tool tool,
									 CodeGenerator generator,
									 Grammar grammar,
									 StringTemplate outputFileST)
		throws IOException
	{
		String fileName =
			generator.getRecognizerFileName(grammar.name, grammar.type);
		generator.write(outputFileST, fileName);
	}

	protected void genRecognizerHeaderFile(Tool tool,
										   CodeGenerator generator,
										   Grammar grammar,
										   StringTemplate headerFileST,
										   String extName) // e.g., ".h"
		throws IOException
	{
		// no header file by default
	}

	protected void performGrammarAnalysis(CodeGenerator generator,
										  Grammar grammar)
	{
		// Build NFAs from the grammar AST
		grammar.buildNFA();

		// Create the DFA predictors for each decision
		grammar.createLookaheadDFAs();
	}

	/** Is scope in @scope::name {action} valid for this kind of grammar?
	 *  Targets like C++ may want to allow new scopes like headerfile or
	 *  some such.  The action names themselves are not policed at the
	 *  moment so targets can add template actions w/o having to recompile
	 *  ANTLR.
	 */
	public boolean isValidActionScope(int grammarType, String scope) {
		switch (grammarType) {
			case Grammar.LEXER :
				if ( scope.equals("lexer") ) {return true;}
				break;
			case Grammar.PARSER :
				if ( scope.equals("parser") ) {return true;}
				break;
			case Grammar.COMBINED :
				if ( scope.equals("parser") ) {return true;}
				if ( scope.equals("lexer") ) {return true;}
				break;
			case Grammar.TREE_PARSER :
				if ( scope.equals("treeparser") ) {return true;}
				break;
		}
		return false;
	}

	/** Target must be able to override the labels used for token types */
	public String getTokenTypeAsTargetLabel(CodeGenerator generator, int ttype) {
		String name = generator.grammar.getTokenDisplayName(ttype);
		// If name is a literal, return the token type instead
		if ( name.charAt(0)=='\'' ) {
			return String.valueOf(ttype);
		}
		return name;
	}

	/** Convert from an ANTLR char literal found in a grammar file to
	 *  an equivalent char literal in the target language.  For most
	 *  languages, this means leaving 'x' as 'x'.  Actually, we need
	 *  to escape '\u000A' so that it doesn't get converted to \n by
	 *  the compiler.  Convert the literal to the char value and then
	 *  to an appropriate target char literal.
	 *
	 *  Expect single quotes around the incoming literal.
	 */
	public String getTargetCharLiteralFromANTLRCharLiteral(
		CodeGenerator generator,
		String literal)
	{
		StringBuffer buf = new StringBuffer();
		buf.append('\'');
		int c = Grammar.getCharValueFromGrammarCharLiteral(literal);
		if ( c<Label.MIN_CHAR_VALUE ) {
			return "'\u0000'";
		}
		if ( c<targetCharValueEscape.length &&
			 targetCharValueEscape[c]!=null )
		{
			buf.append(targetCharValueEscape[c]);
		}
		else if ( Character.UnicodeBlock.of((char)c)==
				  Character.UnicodeBlock.BASIC_LATIN &&
				  !Character.isISOControl((char)c) )
		{
			// normal char
			buf.append((char)c);
		}
		else {
			// must be something unprintable...use \\uXXXX
			// turn on the bit above max "\\uFFFF" value so that we pad with zeros
			// then only take last 4 digits
			String hex = Integer.toHexString(c|0x10000).toUpperCase().substring(1,5);
			buf.append("\\u");
			buf.append(hex);
		}

		buf.append('\'');
		return buf.toString();
	}

	/** Convert from an ANTLR string literal found in a grammar file to
	 *  an equivalent string literal in the target language.  For Java, this
	 *  is the translation 'a\n"' -> "a\n\"".  Expect single quotes
	 *  around the incoming literal.  Just flip the quotes and replace
	 *  double quotes with \"
     * 
     *  Note that we have decided to allow poeple to use '\"' without
     *  penalty, so we must build the target string in a loop as Utils.replae
     *  cannot handle both \" and " without a lot of messing around.
     * 
	 */
	public String getTargetStringLiteralFromANTLRStringLiteral(
		CodeGenerator generator,
		String literal)
	{
        StringBuilder sb = new StringBuilder();
        StringBuffer is = new StringBuffer(literal);
        
        // Opening quote
        //
        sb.append('"');
        
        for (int i = 1; i < is.length() -1; i++) {
            if  (is.charAt(i) == '\\') {
                // Anything escaped is what it is! We assume that
                // people know how to escape characters correctly. However
                // we catch anything that does not need an escape in Java (which
                // is what the default implementation is dealing with and remove 
                // the escape. The C target does this for instance.
                //
                switch (is.charAt(i+1)) {
                    // Pass through any escapes that Java also needs
                    //
                    case    '"':
                    case    'n':
                    case    'r':
                    case    't':
                    case    'b':
                    case    'f':
                    case    '\\':
                    case    'u':    // Assume unnnn
                        sb.append('\\');    // Pass the escape through
                        break;
                    default:
                        // Remove the escape by virtue of not adding it here
                        // Thus \' becomes ' and so on
                        //
                        break;
                }
                
                // Go past the \ character
                //
                i++;
            } else {
                // Chracters that don't need \ in ANTLR 'strings' but do in Java
                //
                if (is.charAt(i) == '"') {
                    // We need to escape " in Java
                    //
                    sb.append('\\');
                }
            }
            // Add in the next character, which may have been escaped
            //
            sb.append(is.charAt(i));   
        }
        
        // Append closing " and return
        //
        sb.append('"');
        
		return sb.toString();
	}

	/** Given a random string of Java unicode chars, return a new string with
	 *  optionally appropriate quote characters for target language and possibly
	 *  with some escaped characters.  For example, if the incoming string has
	 *  actual newline characters, the output of this method would convert them
	 *  to the two char sequence \n for Java, C, C++, ...  The new string has
	 *  double-quotes around it as well.  Example String in memory:
	 *
	 *     a"[newlinechar]b'c[carriagereturnchar]d[tab]e\f
	 *
	 *  would be converted to the valid Java s:
	 *
	 *     "a\"\nb'c\rd\te\\f"
	 *
	 *  or
	 *
	 *     a\"\nb'c\rd\te\\f
	 *
	 *  depending on the quoted arg.
	 */
	public String getTargetStringLiteralFromString(String s, boolean quoted) {
		if ( s==null ) {
			return null;
		}

		StringBuffer buf = new StringBuffer();
		if ( quoted ) {
			buf.append('"');
		}
		for (int i=0; i<s.length(); i++) {
			int c = s.charAt(i);
			if ( c!='\'' && // don't escape single quotes in strings for java
				 c<targetCharValueEscape.length &&
				 targetCharValueEscape[c]!=null )
			{
				buf.append(targetCharValueEscape[c]);
			}
			else {
				buf.append((char)c);
			}
		}
		if ( quoted ) {
			buf.append('"');
		}
		return buf.toString();
	}

	public String getTargetStringLiteralFromString(String s) {
		return getTargetStringLiteralFromString(s, false);
	}

	/** Convert long to 0xNNNNNNNNNNNNNNNN by default for spitting out
	 *  with bitsets.  I.e., convert bytes to hex string.
	 */
	public String getTarget64BitStringFromValue(long word) {
		int numHexDigits = 8*2;
		StringBuffer buf = new StringBuffer(numHexDigits+2);
		buf.append("0x");
		String digits = Long.toHexString(word);
		digits = digits.toUpperCase();
		int padding = numHexDigits - digits.length();
		// pad left with zeros
		for (int i=1; i<=padding; i++) {
			buf.append('0');
		}
		buf.append(digits);
		return buf.toString();
	}

	public String encodeIntAsCharEscape(int v) {
		if ( v<=127 ) {
			return "\\"+Integer.toOctalString(v);
		}
		String hex = Integer.toHexString(v|0x10000).substring(1,5);
		return "\\u"+hex;
	}

	/** Some targets only support ASCII or 8-bit chars/strings.  For example,
	 *  C++ will probably want to return 0xFF here.
	 */
	public int getMaxCharValue(CodeGenerator generator) {
		return Label.MAX_CHAR_VALUE;
	}

	/** Give target a chance to do some postprocessing on actions.
	 *  Python for example will have to fix the indention.
	 */
	public List postProcessAction(List chunks, antlr.Token actionToken) {
		return chunks;
	}

}
