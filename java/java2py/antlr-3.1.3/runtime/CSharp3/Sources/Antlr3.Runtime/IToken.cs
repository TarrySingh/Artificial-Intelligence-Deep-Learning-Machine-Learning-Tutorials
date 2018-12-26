/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008-2009 Sam Harwell, Pixel Mine, Inc.
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

    public interface IToken
    {
        /** <summary>Get the text of the token</summary> */
        string Text
        {
            get;
            set;
        }

        int Type
        {
            get;
            set;
        }

        /** <summary>The line number on which this token was matched; line=1..n</summary> */
        int Line
        {
            get;
            set;
        }

        /** <summary>The index of the first character relative to the beginning of the line 0..n-1</summary> */
        int CharPositionInLine
        {
            get;
            set;
        }

        int Channel
        {
            get;
            set;
        }

        /** <summary>
         *  An index from 0..n-1 of the token object in the input stream.
         *  This must be valid in order to use the ANTLRWorks debugger.
         *  </summary>
         */
        int TokenIndex
        {
            get;
            set;
        }

        /** <summary>
         *  From what character stream was this token created?  You don't have to
         *  implement but it's nice to know where a Token comes from if you have
         *  include files etc... on the input.
         *  </summary>
         */
        ICharStream InputStream
        {
            get;
            set;
        }
    }
}
