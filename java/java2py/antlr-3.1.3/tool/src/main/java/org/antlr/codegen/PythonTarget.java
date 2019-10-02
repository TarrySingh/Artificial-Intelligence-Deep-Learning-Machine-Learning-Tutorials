/*
 [The "BSD licence"]
 Copyright (c) 2005 Martin Traverso
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

/*

Please excuse my obvious lack of Java experience. The code here is probably
full of WTFs - though IMHO Java is the Real WTF(TM) here...

 */

package org.antlr.codegen;
import org.antlr.tool.Grammar;
import java.util.*;

public class PythonTarget extends Target {
    /** Target must be able to override the labels used for token types */
    public String getTokenTypeAsTargetLabel(CodeGenerator generator,
					    int ttype) {
	// use ints for predefined types;
	// <invalid> <EOR> <DOWN> <UP>
	if ( ttype >= 0 && ttype <= 3 ) {
	    return String.valueOf(ttype);
	}

	String name = generator.grammar.getTokenDisplayName(ttype);

	// If name is a literal, return the token type instead
	if ( name.charAt(0)=='\'' ) {
	    return String.valueOf(ttype);
	}

	return name;
    }

    public String getTargetCharLiteralFromANTLRCharLiteral(
            CodeGenerator generator,
            String literal) {
	int c = Grammar.getCharValueFromGrammarCharLiteral(literal);
	return String.valueOf(c);
    }

    private List splitLines(String text) {
		ArrayList l = new ArrayList();
		int idx = 0;

		while ( true ) {
			int eol = text.indexOf("\n", idx);
			if ( eol == -1 ) {
				l.add(text.substring(idx));
				break;
			}
			else {
				l.add(text.substring(idx, eol+1));
				idx = eol+1;
			}
		}

		return l;
    }

    public List postProcessAction(List chunks, antlr.Token actionToken) {
		/* TODO
		   - check for and report TAB usage
		 */

		//System.out.println("\n*** Action at " + actionToken.getLine() + ":" + actionToken.getColumn());

		/* First I create a new list of chunks. String chunks are splitted into
		   lines and some whitespace my be added at the beginning.

		   As a result I get a list of chunks
		   - where the first line starts at column 0
		   - where every LF is at the end of a string chunk
		*/

		List nChunks = new ArrayList();
		for (int i = 0; i < chunks.size(); i++) {
			Object chunk = chunks.get(i);

			if ( chunk instanceof String ) {
				String text = (String)chunks.get(i);
				if ( nChunks.size() == 0 && actionToken.getColumn() > 0 ) {
					// first chunk and some 'virtual' WS at beginning
					// prepend to this chunk

					String ws = "";
					for ( int j = 0 ; j < actionToken.getColumn() ; j++ ) {
						ws += " ";
					}
					text = ws + text;
				}

				List parts = splitLines(text);
				for ( int j = 0 ; j < parts.size() ; j++ ) {
					chunk = parts.get(j);
					nChunks.add(chunk);
				}
			}
			else {
				if ( nChunks.size() == 0 && actionToken.getColumn() > 0 ) {
					// first chunk and some 'virtual' WS at beginning
					// add as a chunk of its own

					String ws = "";
					for ( int j = 0 ; j < actionToken.getColumn() ; j++ ) {
						ws += " ";
					}
					nChunks.add(ws);
				}

				nChunks.add(chunk);
			}
		}

		int lineNo = actionToken.getLine();
		int col = 0;

		// strip trailing empty lines
		int lastChunk = nChunks.size() - 1;
		while ( lastChunk > 0
				&& nChunks.get(lastChunk) instanceof String
				&& ((String)nChunks.get(lastChunk)).trim().length() == 0 )
			lastChunk--;

		// string leading empty lines
		int firstChunk = 0;
		while ( firstChunk <= lastChunk
				&& nChunks.get(firstChunk) instanceof String
				&& ((String)nChunks.get(firstChunk)).trim().length() == 0
				&& ((String)nChunks.get(firstChunk)).endsWith("\n") ) {
			lineNo++;
			firstChunk++;
		}

		int indent = -1;
		for ( int i = firstChunk ; i <= lastChunk ; i++ ) {
			Object chunk = nChunks.get(i);

			//System.out.println(lineNo + ":" + col + " " + quote(chunk.toString()));

			if ( chunk instanceof String ) {
				String text = (String)chunk;

				if ( col == 0 ) {
					if ( indent == -1 ) {
						// first non-blank line
						// count number of leading whitespaces

						indent = 0;
						for ( int j = 0; j < text.length(); j++ ) {
							if ( !Character.isWhitespace(text.charAt(j)) )
								break;
			
							indent++;
						}
					}

					if ( text.length() >= indent ) {
						int j;
						for ( j = 0; j < indent ; j++ ) {
							if ( !Character.isWhitespace(text.charAt(j)) ) {
								// should do real error reporting here...
								System.err.println("Warning: badly indented line " + lineNo + " in action:");
								System.err.println(text);
								break;
							}
						}

						nChunks.set(i, text.substring(j));
					}
					else if ( text.trim().length() > 0 ) {
						// should do real error reporting here...
						System.err.println("Warning: badly indented line " + lineNo + " in action:");
						System.err.println(text);
					}
				}

				if ( text.endsWith("\n") ) {
					lineNo++;
					col = 0;
				}
				else {
					col += text.length();
				}
			}
			else {
				// not really correct, but all I need is col to increment...
				col += 1;
			}
		}

		return nChunks;
    }
}
