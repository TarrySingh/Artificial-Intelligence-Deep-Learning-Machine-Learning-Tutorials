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
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.tool.Grammar;

public class ActionScriptTarget extends Target {

    public String getTargetCharLiteralFromANTLRCharLiteral(
            CodeGenerator generator,
            String literal) {

        int c = Grammar.getCharValueFromGrammarCharLiteral(literal);
        return String.valueOf(c);
    }

    public String getTokenTypeAsTargetLabel(CodeGenerator generator,
                                            int ttype) {
        // use ints for predefined types;
        // <invalid> <EOR> <DOWN> <UP>
        if (ttype >= 0 && ttype <= 3) {
            return String.valueOf(ttype);
        }

        String name = generator.grammar.getTokenDisplayName(ttype);

        // If name is a literal, return the token type instead
        if (name.charAt(0) == '\'') {
            return String.valueOf(ttype);
        }

        return name;
    }

    /**
     * ActionScript doesn't support Unicode String literals that are considered "illegal"
     * or are in the surrogate pair ranges.  For example "/uffff" will not encode properly
     * nor will "/ud800".  To keep things as compact as possible we use the following encoding
     * if the int is below 255, we encode as hex literal
     * If the int is between 255 and 0x7fff we use a single unicode literal with the value
     * If the int is above 0x7fff, we use a unicode literal of 0x80hh, where hh is the high-order
     * bits followed by \xll where ll is the lower order bits of a 16-bit number.
     *
     * Ideally this should be improved at a future date.  The most optimal way to encode this
     * may be a compressed AMF encoding that is embedded using an Embed tag in ActionScript.
     *
     * @param v
     * @return
     */
    public String encodeIntAsCharEscape(int v) {
        // encode as hex
        if ( v<=255 ) {
			return "\\x"+ Integer.toHexString(v|0x100).substring(1,3);
		}
        if (v <= 0x7fff) {
            String hex = Integer.toHexString(v|0x10000).substring(1,5);
		    return "\\u"+hex;
        }
        if (v > 0xffff) {
            System.err.println("Warning: character literal out of range for ActionScript target " + v);
            return "";
        }
        StringBuffer buf = new StringBuffer("\\u80");
        buf.append(Integer.toHexString((v >> 8) | 0x100).substring(1, 3)); // high - order bits
        buf.append("\\x");
        buf.append(Integer.toHexString((v & 0xff) | 0x100).substring(1, 3)); // low -order bits
        return buf.toString();
    }

    /** Convert long to two 32-bit numbers separted by a comma.
     *  ActionScript does not support 64-bit numbers, so we need to break
     *  the number into two 32-bit literals to give to the Bit.  A number like
     *  0xHHHHHHHHLLLLLLLL is broken into the following string:
     *  "0xLLLLLLLL, 0xHHHHHHHH"
	 *  Note that the low order bits are first, followed by the high order bits.
     *  This is to match how the BitSet constructor works, where the bits are
     *  passed in in 32-bit chunks with low-order bits coming first.
	 */
	public String getTarget64BitStringFromValue(long word) {
		StringBuffer buf = new StringBuffer(22); // enough for the two "0x", "," and " "
		buf.append("0x");
        writeHexWithPadding(buf, Integer.toHexString((int)(word & 0x00000000ffffffffL)));
        buf.append(", 0x");
        writeHexWithPadding(buf, Integer.toHexString((int)(word >> 32)));

        return buf.toString();
	}

    private void writeHexWithPadding(StringBuffer buf, String digits) {
       digits = digits.toUpperCase();
		int padding = 8 - digits.length();
		// pad left with zeros
		for (int i=1; i<=padding; i++) {
			buf.append('0');
		}
		buf.append(digits);
    }

    protected StringTemplate chooseWhereCyclicDFAsGo(Tool tool,
                                                     CodeGenerator generator,
                                                     Grammar grammar,
                                                     StringTemplate recognizerST,
                                                     StringTemplate cyclicDFAST) {
        return recognizerST;
    }
}

