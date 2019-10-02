/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008 Sam Harwell, Pixel Mine, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace Antlr.Runtime
{
    using System;

    /** <summary>An extra token while parsing a TokenStream</summary> */
    [Serializable]
    public class UnwantedTokenException : MismatchedTokenException
    {
        /** <summary>Used for remote debugger deserialization</summary> */
        public UnwantedTokenException()
        {
        }

        public UnwantedTokenException( int expecting, IIntStream input )
            : base( expecting, input )
        {
        }

        public virtual IToken UnexpectedToken
        {
            get
            {
                return token;
            }
        }

        public override string ToString()
        {
            //int unexpectedType = getUnexpectedType();
            //string unexpected = ( tokenNames != null && unexpectedType >= 0 && unexpectedType < tokenNames.Length ) ? tokenNames[unexpectedType] : unexpectedType.ToString();
            string expected = ( tokenNames != null && expecting >= 0 && expecting < tokenNames.Length ) ? tokenNames[expecting] : expecting.ToString();

            String exp = ", expected " + expected;
            if ( expecting == TokenConstants.INVALID_TOKEN_TYPE )
            {
                exp = "";
            }
            if ( token == null )
            {
                return "UnwantedTokenException(found=" + null + exp + ")";
            }
            return "UnwantedTokenException(found=" + token.Text + exp + ")";
        }
    }
}
