/*
 [The "BSD licence"]
 Copyright (c) 2007 Ronald Blaschke
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

import org.antlr.analysis.Label;
import org.antlr.tool.AttributeScope;
import org.antlr.tool.Grammar;
import org.antlr.tool.RuleLabelScope;

public class Perl5Target extends Target {
    public Perl5Target() {
        targetCharValueEscape['$'] = "\\$";
        targetCharValueEscape['@'] = "\\@";
        targetCharValueEscape['%'] = "\\%";
        AttributeScope.tokenScope.addAttribute("self", null);
        RuleLabelScope.predefinedLexerRulePropertiesScope.addAttribute("self", null);
    }

    public String getTargetCharLiteralFromANTLRCharLiteral(final CodeGenerator generator,
                                                           final String literal) {
        final StringBuffer buf = new StringBuffer(10);

        final int c = Grammar.getCharValueFromGrammarCharLiteral(literal);
        if (c < Label.MIN_CHAR_VALUE) {
            buf.append("\\x{0000}");
        } else if (c < targetCharValueEscape.length &&
                targetCharValueEscape[c] != null) {
            buf.append(targetCharValueEscape[c]);
        } else if (Character.UnicodeBlock.of((char) c) ==
                Character.UnicodeBlock.BASIC_LATIN &&
                !Character.isISOControl((char) c)) {
            // normal char
            buf.append((char) c);
        } else {
            // must be something unprintable...use \\uXXXX
            // turn on the bit above max "\\uFFFF" value so that we pad with zeros
            // then only take last 4 digits
            String hex = Integer.toHexString(c | 0x10000).toUpperCase().substring(1, 5);
            buf.append("\\x{");
            buf.append(hex);
            buf.append("}");
        }

        if (buf.indexOf("\\") == -1) {
            // no need for interpolation, use single quotes
            buf.insert(0, '\'');
            buf.append('\'');
        } else {
            // need string interpolation
            buf.insert(0, '\"');
            buf.append('\"');
        }

        return buf.toString();
    }

    public String encodeIntAsCharEscape(final int v) {
        final int intValue;
        if ((v & 0x8000) == 0) {
            intValue = v;
        } else {
            intValue = -(0x10000 - v);
        }

        return String.valueOf(intValue);
    }
}
