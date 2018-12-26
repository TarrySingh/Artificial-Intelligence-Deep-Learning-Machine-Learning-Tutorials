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

    /** <summary>
     *  Queues up nodes matched on left side of -> in a tree parser. This is
     *  the analog of RewriteRuleTokenStream for normal parsers.
     *  </summary>
     */
    [System.Serializable]
    public class RewriteRuleNodeStream : RewriteRuleElementStream
    {

        public RewriteRuleNodeStream( ITreeAdaptor adaptor, string elementDescription )
            : base( adaptor, elementDescription )
        {
        }

        /** <summary>Create a stream with one element</summary> */
        public RewriteRuleNodeStream( ITreeAdaptor adaptor, string elementDescription, object oneElement )
            : base( adaptor, elementDescription, oneElement )
        {
        }

        /** <summary>Create a stream, but feed off an existing list</summary> */
        public RewriteRuleNodeStream( ITreeAdaptor adaptor, string elementDescription, IList elements )
            : base( adaptor, elementDescription, elements )
        {
        }

        public virtual object NextNode()
        {
            return _Next();
        }

        protected override object ToTree( object el )
        {
            return adaptor.DupNode( el );
        }

        protected override object Dup( object el )
        {
            // we dup every node, so don't have to worry about calling dup; short-
            // circuited next() so it doesn't call.
            throw new NotSupportedException( "dup can't be called for a node stream." );
        }
    }
}
