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

package org.antlr.codegen;

public class RubyTarget
        extends Target
{
    public String getTargetCharLiteralFromANTLRCharLiteral(
            CodeGenerator generator,
            String literal)
    {
        literal = literal.substring(1, literal.length() - 1);

        String result = "?";

        if (literal.equals("\\")) {
            result += "\\\\";
        }
        else if (literal.equals(" ")) {
            result += "\\s";
        }
        else if (literal.startsWith("\\u")) {
            result = "0x" + literal.substring(2);
        }
        else {
            result += literal;
        }

        return result;
    }

    public int getMaxCharValue(CodeGenerator generator)
    {
        // we don't support unicode, yet.
        return 0xFF;
    }

    public String getTokenTypeAsTargetLabel(CodeGenerator generator, int ttype)
    {
        String name = generator.grammar.getTokenDisplayName(ttype);
        // If name is a literal, return the token type instead
        if ( name.charAt(0)=='\'' ) {
            return generator.grammar.computeTokenNameFromLiteral(ttype, name);
        }
        return name;
    }
}
