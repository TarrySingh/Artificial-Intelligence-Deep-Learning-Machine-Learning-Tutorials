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

namespace Antlr.Runtime.Tree
{
    using IList = System.Collections.IList;
    using NotSupportedException = System.NotSupportedException;

    [System.Serializable]
    public class RewriteRuleTokenStream : RewriteRuleElementStream
    {

        public RewriteRuleTokenStream( ITreeAdaptor adaptor, string elementDescription )
            : base( adaptor, elementDescription )
        {
        }

        /** <summary>Create a stream with one element</summary> */
        public RewriteRuleTokenStream( ITreeAdaptor adaptor, string elementDescription, object oneElement )
            : base( adaptor, elementDescription, oneElement )
        {
        }

        /** <summary>Create a stream, but feed off an existing list</summary> */
        public RewriteRuleTokenStream( ITreeAdaptor adaptor, string elementDescription, IList elements )
            : base( adaptor, elementDescription, elements )
        {
        }

        /** <summary>Get next token from stream and make a node for it</summary> */
        public virtual object NextNode()
        {
            IToken t = (IToken)_Next();
            return adaptor.Create( t );
        }

        public virtual IToken NextToken()
        {
            return (IToken)_Next();
        }

        /** <summary>
         *  Don't convert to a tree unless they explicitly call nextTree.
         *  This way we can do hetero tree nodes in rewrite.
         *  </summary>
         */
        protected override object ToTree( object el )
        {
            return el;
        }

        protected override object Dup( object el )
        {
            throw new NotSupportedException( "dup can't be called for a token stream." );
        }
    }
}
